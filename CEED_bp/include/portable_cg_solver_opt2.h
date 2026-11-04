#ifndef portable_cg_solver_opt2_h
#define portable_cg_solver_opt2_h

#include <deal.II/base/config.h>
#include <deal.II/base/array_view.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector_memory.h>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace PortableOpt2
{
  using namespace dealii;

  enum class CGMode
  {
    /**
     * Exact classical unpreconditioned CG recurrence (identical residual
     * sequence and iteration count to dealii::SolverCG), fusing vector updates
     * and reductions on compute_exec to cut 7 GPU stream syncs -> 2 syncs/iter.
     */
    FusedExact,

    /**
     * GPU Kokkos port of deal.II's Host-only single-Allreduce IterationWorker
     * (solver_cg.h:862-912): 1 GPU reduction + 1 stream sync + 1 MPI_Allreduce
     * (4 doubles) + 1 fused (x, r, p) update kernel per iteration.
     */
    FusedSingleAllreduce,

    /**
     * Delegates directly to unmodified dealii::SolverCG<VectorType>.
     */
    DealII
  };

  namespace internal
  {
    template <typename MatrixType, typename ExecType, typename VectorType>
    class HasVmultAsync
    {
    private:
      template <typename M>
      static auto
      test(int) -> decltype(
        std::declval<const M &>().vmult_async(std::declval<const ExecType &>(),
                                              std::declval<VectorType &>(),
                                              std::declval<const VectorType &>()),
        std::declval<const M &>().get_compute_exec(),
        std::true_type{});

      template <typename>
      static std::false_type
      test(...);

    public:
      static constexpr bool value = decltype(test<MatrixType>(0))::value;
    };

    struct QuadScalarSum
    {
      double pv = 0.0;
      double vv = 0.0;
      double rv = 0.0;
      double rr = 0.0;

      KOKKOS_INLINE_FUNCTION void
      operator+=(const QuadScalarSum &rhs)
      {
        pv += rhs.pv;
        vv += rhs.vv;
        rv += rhs.rv;
        rr += rhs.rr;
      }

#if defined(KOKKOS_ENABLE_DEPRECATED_CODE_4)
      KOKKOS_INLINE_FUNCTION void
      operator+=(const volatile QuadScalarSum &rhs) volatile
      {
        pv += rhs.pv;
        vv += rhs.vv;
        rv += rhs.rv;
        rr += rhs.rr;
      }
#endif
    };

    template <typename Number>
    struct NormSqrFunctor
    {
      const Number *x;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, double &sum) const
      {
        const double xi = static_cast<double>(x[i]);
        sum += xi * xi;
      }
    };

    template <typename Number>
    struct CopyRPAndNormFunctor
    {
      Number       *r;
      Number       *p;
      const Number *b;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, double &sum) const
      {
        const Number bi = b[i];
        r[i]            = bi;
        p[i]            = bi;
        const double bd = static_cast<double>(bi);
        sum += bd * bd;
      }
    };

    template <typename Number>
    struct ResidualInitRPAndNormFunctor
    {
      Number       *r;
      Number       *p;
      const Number *b;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, double &sum) const
      {
        const Number ri = b[i] - r[i];
        r[i]            = ri;
        p[i]            = ri;
        const double rd = static_cast<double>(ri);
        sum += rd * rd;
      }
    };

    template <typename Number>
    struct UpdateSearchDirFunctor
    {
      Number       *p;
      const Number *r;
      Number        beta;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i) const
      {
        p[i] = r[i] + beta * p[i];
      }
    };

    template <typename Number>
    struct DotProductFunctor
    {
      const Number *p;
      const Number *v;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, double &sum) const
      {
        sum += static_cast<double>(p[i]) * static_cast<double>(v[i]);
      }
    };

    template <typename Number>
    struct FusedUpdateXRAndNormFunctor
    {
      Number       *x;
      Number       *r;
      const Number *p;
      const Number *v;
      Number        alpha;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, double &sum) const
      {
        x[i] += alpha * p[i];
        const Number r_new = r[i] - alpha * v[i];
        r[i]               = r_new;
        const double rd    = static_cast<double>(r_new);
        sum += rd * rd;
      }
    };

    template <typename Number>
    struct FusedDeferredXRPUpdateFunctor
    {
      Number       *x;
      Number       *r;
      Number       *p;
      const Number *v;
      Number        prev_alpha;
      Number        beta;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i) const
      {
        const Number p_old = p[i];
        const Number r_new = r[i] - prev_alpha * v[i];
        x[i] += prev_alpha * p_old;
        r[i] = r_new;
        p[i] = r_new + beta * p_old;
      }
    };

    template <typename Number>
    struct FinalizeXRUpdateFunctor
    {
      Number       *x;
      Number       *r;
      const Number *p;
      const Number *v;
      Number        alpha;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i) const
      {
        x[i] += alpha * p[i];
        r[i] -= alpha * v[i];
      }
    };

    template <typename Number>
    struct FusedQuadDotFunctor
    {
      const Number *p;
      const Number *v;
      const Number *r;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i, QuadScalarSum &acc) const
      {
        const double pi = static_cast<double>(p[i]);
        const double vi = static_cast<double>(v[i]);
        const double ri = static_cast<double>(r[i]);
        acc.pv += pi * vi;
        acc.vv += vi * vi;
        acc.rv += ri * vi;
        acc.rr += ri * ri;
      }
    };

  } // namespace internal

  /**
   * Local header-only adaptation of dealii::SolverCG<VectorType> preserving
   * deal.II's SolverBase<VectorType>, SolverControl, and VectorMemory<VectorType>
   * structure without modifying installed deal.II headers or libraries.
   */
  template <typename VectorType =
              LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
  class SolverCGOpt2 : public SolverBase<VectorType>
  {
  public:
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;

    struct AdditionalData
    {
      explicit AdditionalData(const CGMode mode = CGMode::FusedExact)
        : mode(mode)
      {}

      CGMode mode = CGMode::FusedExact;
    };

    SolverCGOpt2(SolverControl            &cn,
                 VectorMemory<VectorType> &mem,
                 const AdditionalData     &data = AdditionalData())
      : SolverBase<VectorType>(cn, mem)
      , solver_control_ref(cn)
      , additional_data(data)
    {}

    explicit SolverCGOpt2(SolverControl        &cn,
                          const AdditionalData &data = AdditionalData())
      : SolverBase<VectorType>(cn)
      , solver_control_ref(cn)
      , additional_data(data)
    {}

    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType         &A,
          VectorType               &x,
          const VectorType         &b,
          const PreconditionerType &preconditioner)
    {
      if constexpr (!std::is_same_v<PreconditionerType, PreconditionIdentity>)
        {
          SolverCG<VectorType> fallback(solver_control_ref, this->memory);
          fallback.solve(A, x, b, preconditioner);
          return;
        }
      else
        {
          if (additional_data.mode == CGMode::DealII)
            {
              SolverCG<VectorType> dealii_solver(solver_control_ref,
                                                 this->memory);
              dealii_solver.solve(A, x, b, preconditioner);
              return;
            }

          if (additional_data.mode == CGMode::FusedSingleAllreduce)
            solve_fused_single_allreduce(A, x, b);
          else
            solve_fused_exact(A, x, b);
        }
    }

  private:
    template <typename MatrixType>
    static ExecutionSpace
    select_exec(const MatrixType &A)
    {
      if constexpr (internal::HasVmultAsync<MatrixType,
                                            ExecutionSpace,
                                            VectorType>::value)
        return A.get_compute_exec();
      else
        return ExecutionSpace{};
    }

    template <typename MatrixType>
    static void
    apply_operator_async(const MatrixType     &A,
                         const ExecutionSpace &exec,
                         VectorType           &dst,
                         const VectorType     &src)
    {
      if constexpr (internal::HasVmultAsync<MatrixType,
                                            ExecutionSpace,
                                            VectorType>::value)
        A.vmult_async(exec, dst, src);
      else
        A.vmult(dst, src);
    }

    static double
    allreduce_scalar(const double local_val, const VectorType &vec)
    {
      const auto &part = vec.get_partitioner();
      if (part && part->n_mpi_processes() > 1u)
        return Utilities::MPI::sum(local_val, part->get_mpi_communicator());
      return local_val;
    }

    template <typename MatrixType>
    double
    startup_vectors(const MatrixType     &A,
                    const ExecutionSpace &exec,
                    VectorType           &x,
                    const VectorType     &b,
                    VectorType           &r,
                    VectorType           &p,
                    VectorType           &v)
    {
      using Number = typename VectorType::value_type;

      r.reinit(x, /*omit_zeroing_entries=*/true);
      p.reinit(x, /*omit_zeroing_entries=*/true);
      v.reinit(x, /*omit_zeroing_entries=*/true);

      const unsigned int n_local = x.locally_owned_size();

      double local_x_norm2 = 0.0;
      if (n_local > 0u)
        {
          Kokkos::parallel_reduce(
            "PortableOpt2::CG::x_norm2",
            Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
            internal::NormSqrFunctor<Number>{x.get_values()},
            local_x_norm2);
        }
      else
        {
          exec.fence();
        }

      const double global_x_norm2 = allreduce_scalar(local_x_norm2, x);

      double local_r_norm2 = 0.0;
      if (global_x_norm2 == 0.0)
        {
          if (n_local > 0u)
            {
              Kokkos::parallel_reduce(
                "PortableOpt2::CG::init_rp_zero_x",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::CopyRPAndNormFunctor<Number>{r.get_values(),
                                                       p.get_values(),
                                                       b.get_values()},
                local_r_norm2);
            }
          else
            {
              exec.fence();
            }
        }
      else
        {
          apply_operator_async(A, exec, r, x);
          if (n_local > 0u)
            {
              Kokkos::parallel_reduce(
                "PortableOpt2::CG::init_rp_nonzero_x",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::ResidualInitRPAndNormFunctor<Number>{r.get_values(),
                                                               p.get_values(),
                                                               b.get_values()},
                local_r_norm2);
            }
          else
            {
              exec.fence();
            }
        }

      return allreduce_scalar(local_r_norm2, x);
    }

    template <typename MatrixType>
    void
    solve_fused_exact(const MatrixType &A, VectorType &x, const VectorType &b)
    {
      using Number = typename VectorType::value_type;

      typename VectorMemory<VectorType>::Pointer r_ptr(this->memory);
      typename VectorMemory<VectorType>::Pointer p_ptr(this->memory);
      typename VectorMemory<VectorType>::Pointer v_ptr(this->memory);
      VectorType &r = *r_ptr;
      VectorType &p = *p_ptr;
      VectorType &v = *v_ptr;

      const ExecutionSpace exec    = select_exec(A);
      const unsigned int   n_local = x.locally_owned_size();

      double rr            = startup_vectors(A, exec, x, b, r, p, v);
      double residual_norm = std::sqrt(std::abs(rr));

      SolverControl::State state = this->iteration_status(0u, residual_norm, x);
      if (state != SolverControl::iterate)
        return;

      unsigned int it   = 0u;
      double       beta = 0.0;

      while (state == SolverControl::iterate)
        {
          ++it;

          // 1. Update search direction p = r + beta * p (iter 1 already has p == r!)
          //    and chain directly into vmult_async with ZERO host sync
          if (it > 1u && n_local > 0u)
            {
              Kokkos::parallel_for(
                "PortableOpt2::CG::update_p",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::UpdateSearchDirFunctor<Number>{
                  p.get_values(), r.get_values(), static_cast<Number>(beta)});
            }

          apply_operator_async(A, exec, v, p);

          // 2. Compute p * v (Sync #1 + Allreduce #1)
          double local_pv = 0.0;
          if (n_local > 0u)
            {
              Kokkos::parallel_reduce(
                "PortableOpt2::CG::dot_pv",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::DotProductFunctor<Number>{p.get_values(),
                                                    v.get_values()},
                local_pv);
            }
          else
            {
              exec.fence();
            }

          const double pv = allreduce_scalar(local_pv, x);
          Assert(std::abs(pv) != 0.0, ExcDivideByZero());
          const double alpha = rr / pv;

          // 3. Fused x += alpha * p, r -= alpha * v, and ||r||_2^2 in ONE pass
          //    (Sync #2 + Allreduce #2)
          double local_rr_new = 0.0;
          if (n_local > 0u)
            {
              Kokkos::parallel_reduce(
                "PortableOpt2::CG::fused_update_xr_norm",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::FusedUpdateXRAndNormFunctor<Number>{
                  x.get_values(),
                  r.get_values(),
                  p.get_values(),
                  v.get_values(),
                  static_cast<Number>(alpha)},
                local_rr_new);
            }
          else
            {
              exec.fence();
            }

          const double rr_new = allreduce_scalar(local_rr_new, x);
          residual_norm       = std::sqrt(std::abs(rr_new));
          beta                = rr_new / rr;
          rr                  = rr_new;

          state = this->iteration_status(it, residual_norm, x);
        }

      AssertThrow(state == SolverControl::success,
                  SolverControl::NoConvergence(it, residual_norm));
    }

    template <typename MatrixType>
    void
    solve_fused_single_allreduce(const MatrixType &A,
                                 VectorType       &x,
                                 const VectorType &b)
    {
      using Number = typename VectorType::value_type;

      typename VectorMemory<VectorType>::Pointer r_ptr(this->memory);
      typename VectorMemory<VectorType>::Pointer p_ptr(this->memory);
      typename VectorMemory<VectorType>::Pointer v_ptr(this->memory);
      VectorType &r = *r_ptr;
      VectorType &p = *p_ptr;
      VectorType &v = *v_ptr;

      const ExecutionSpace exec    = select_exec(A);
      const unsigned int   n_local = x.locally_owned_size();
      const auto          &part    = x.get_partitioner();
      const bool           is_mpi  = (part && part->n_mpi_processes() > 1u);

      double rr            = startup_vectors(A, exec, x, b, r, p, v);
      double residual_norm = std::sqrt(std::abs(rr));

      SolverControl::State state = this->iteration_status(0u, residual_norm, x);
      if (state != SolverControl::iterate)
        return;

      unsigned int it         = 0u;
      double       prev_alpha = 0.0;
      double       beta       = 0.0;

      while (state == SolverControl::iterate)
        {
          ++it;

          // Deferred fused update of (x, r, p) from previous iteration + vmult_async
          if (it > 1u && n_local > 0u)
            {
              Kokkos::parallel_for(
                "PortableOpt2::CG::fused_xrp_update",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::FusedDeferredXRPUpdateFunctor<Number>{
                  x.get_values(),
                  r.get_values(),
                  p.get_values(),
                  v.get_values(),
                  static_cast<Number>(prev_alpha),
                  static_cast<Number>(beta)});
            }

          apply_operator_async(A, exec, v, p);

          // Single GPU reduction + single 4-double MPI_Allreduce per iteration
          internal::QuadScalarSum local_sums{};
          if (n_local > 0u)
            {
              Kokkos::parallel_reduce(
                "PortableOpt2::CG::quad_dot",
                Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                internal::FusedQuadDotFunctor<Number>{
                  p.get_values(), v.get_values(), r.get_values()},
                local_sums);
            }
          else
            {
              exec.fence();
            }

          std::array<double, 4> sums = {{local_sums.pv,
                                         local_sums.vv,
                                         local_sums.rv,
                                         local_sums.rr}};
          if (is_mpi)
            {
              Utilities::MPI::sum(
                ArrayView<const double>(sums.data(), 4),
                part->get_mpi_communicator(),
                ArrayView<double>(sums.data(), 4));
            }

          const double pv      = sums[0];
          const double vv      = sums[1];
          const double rv      = sums[2];
          const double rr_curr = sums[3];
          Assert(std::abs(pv) != 0.0, ExcDivideByZero());

          const double alpha = rr_curr / pv;
          const double rr_next_est =
            std::abs(rr_curr + alpha * (-2.0 * rv + alpha * vv));
          residual_norm = std::sqrt(rr_next_est);

          state = this->iteration_status(it, residual_norm, x);
          if (state == SolverControl::iterate)
            {
              prev_alpha = alpha;
              beta       = rr_next_est / rr_curr;
            }
          else
            {
              // Apply final iterate update x += alpha * p before returning
              if (n_local > 0u)
                {
                  Kokkos::parallel_for(
                    "PortableOpt2::CG::finalize_xr",
                    Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_local),
                    internal::FinalizeXRUpdateFunctor<Number>{
                      x.get_values(),
                      r.get_values(),
                      p.get_values(),
                      v.get_values(),
                      static_cast<Number>(alpha)});
                }
              exec.fence();
            }
        }

      AssertThrow(state == SolverControl::success,
                  SolverControl::NoConvergence(it, residual_norm));
    }

    SolverControl &solver_control_ref;
    AdditionalData additional_data;
  };

} // namespace PortableOpt2

#endif // portable_cg_solver_opt2_h
