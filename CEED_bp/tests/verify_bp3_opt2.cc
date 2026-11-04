#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/read_write_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "portable_cg_solver_opt2.h"
#include "portable_laplace_operator_opt2.h"

using namespace dealii;

namespace
{
  template <int dim>
  class CurvedShearedManifold : public ChartManifold<dim, dim, dim>
  {
  public:
    static Point<dim>
    map_forward(const Point<dim> &chart_point)
    {
      static_assert(dim == 3, "CurvedShearedManifold requires dim == 3.");
      const double xi   = chart_point[0];
      const double eta  = chart_point[1];
      const double zeta = chart_point[2];

      const double z = 0.7 * zeta;
      const double y = 1.1 * eta + 0.12 * std::sin(numbers::PI * zeta);
      const double x = 2.3 * xi + 0.18 * y +
                       0.15 * std::sin(numbers::PI * eta) *
                         std::cos(numbers::PI * zeta);
      return Point<dim>(x, y, z);
    }

    static Point<dim>
    map_backward(const Point<dim> &space_point)
    {
      static_assert(dim == 3, "CurvedShearedManifold requires dim == 3.");
      const double x = space_point[0];
      const double y = space_point[1];
      const double z = space_point[2];

      const double zeta = z / 0.7;
      const double eta  = (y - 0.12 * std::sin(numbers::PI * zeta)) / 1.1;
      const double xi =
        (x - 0.18 * y -
         0.15 * std::sin(numbers::PI * eta) * std::cos(numbers::PI * zeta)) /
        2.3;
      return Point<dim>(xi, eta, zeta);
    }

    Point<dim>
    push_forward(const Point<dim> &chart_point) const override
    {
      return map_forward(chart_point);
    }

    Point<dim>
    pull_back(const Point<dim> &space_point) const override
    {
      return map_backward(space_point);
    }

    std::unique_ptr<Manifold<dim, dim>>
    clone() const override
    {
      return std::make_unique<CurvedShearedManifold<dim>>();
    }
  };

  template <int dim, int fe_degree, int nq, typename Number>
  struct LocalLaplaceReferenceFunctor
  {
    static constexpr unsigned int n_q_points = Utilities::pow(nq, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
               const Portable::DeviceVector<Number>                   &src,
               Portable::DeviceVector<Number>                         &dst) const
    {
      Portable::FEEvaluation<dim, fe_degree, nq, 1, Number> phi(data);
      phi.read_dof_values(src);
      phi.evaluate(EvaluationFlags::gradients);
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(data->team_member, n_q_points),
        [&](const int &q) { phi.submit_gradient(phi.get_gradient(q), q); });
      data->team_member.team_barrier();
      phi.integrate(EvaluationFlags::gradients);
      phi.distribute_local_to_global(dst);
    }
  };

  template <typename Number>
  void
  copy_owned_host_to_device(
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &src_h,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>    &dst_d)
  {
    AssertDimension(src_h.locally_owned_size(), dst_d.locally_owned_size());
    dst_d = Number(0);
    const unsigned int n_owned = src_h.locally_owned_size();
    if (n_owned > 0u)
      {
        Kokkos::deep_copy(
          Kokkos::View<Number *, typename MemorySpace::Default::kokkos_space>(
            dst_d.get_values(), n_owned),
          Kokkos::View<const Number *, Kokkos::HostSpace>(src_h.get_values(),
                                                          n_owned));
      }
    Kokkos::fence();
  }

  template <typename Number>
  void
  copy_owned_device_to_host(
    const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
                                                                  &src_d,
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> &dst_h)
  {
    AssertDimension(src_d.locally_owned_size(), dst_h.locally_owned_size());
    dst_h = Number(0);
    const unsigned int n_owned = src_d.locally_owned_size();
    if (n_owned > 0u)
      {
        Kokkos::deep_copy(
          Kokkos::View<Number *, Kokkos::HostSpace>(dst_h.get_values(),
                                                    n_owned),
          Kokkos::View<const Number *,
                       typename MemorySpace::Default::kokkos_space>(
            src_d.get_values(), n_owned));
      }
  }

  template <int dim, int fe_degree, int nq, typename Number>
  class DealIIReferenceLaplaceOperator
  {
  public:
    DealIIReferenceLaplaceOperator(const DoFHandler<dim>           &dof_handler,
                                   const AffineConstraints<Number> &constraints)
      : dof_handler_ref(dof_handler)
      , constraints_ref(constraints)
      , use_host_mpi_fallback(
          Utilities::MPI::n_mpi_processes(dof_handler.get_mpi_communicator()) >
            1 &&
          !PortableOpt2::query_device_aware_mpi())
    {
      if (use_host_mpi_fallback)
        {
          const IndexSet locally_relevant =
            DoFTools::extract_locally_relevant_dofs(dof_handler);
          host_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
            dof_handler.locally_owned_dofs(),
            locally_relevant,
            dof_handler.get_mpi_communicator());
        }
      else
        {
          const MappingQ<dim> mapping(fe_degree);
          typename Portable::MatrixFree<dim, Number>::AdditionalData
            additional_data;
          additional_data.mapping_update_flags =
            update_gradients | update_JxW_values;
          additional_data.overlap_communication_computation = false;
          const QGauss<1> quad(nq);
          matrix_free.reinit(
            mapping, dof_handler, constraints, quad, additional_data);
        }
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec)
      const
    {
      if (use_host_mpi_fallback)
        vec.reinit(host_partitioner);
      else
        matrix_free.initialize_dof_vector(vec);
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
            &src) const
    {
      if (!use_host_mpi_fallback)
        {
          LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
            src_masked;
          matrix_free.initialize_dof_vector(src_masked);
          src_masked = src;
          matrix_free.set_constrained_values(Number(0), src_masked);

          dst = Number(0);
          LocalLaplaceReferenceFunctor<dim, fe_degree, nq, Number> functor;
          matrix_free.cell_loop(functor, src_masked, dst);
          matrix_free.copy_constrained_values(src, dst);
          Kokkos::fence();
          return;
        }

      // Host FEValues reference assembly when deal.II's prebuilt device MPI
      // path cannot be used with a non-GPU-aware MPI library at runtime.
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> src_host(
        host_partitioner);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> dst_host(
        host_partitioner);

      copy_owned_device_to_host(src, src_host);
      src_host.update_ghost_values();
      dst_host = Number(0);

      const MappingQ<dim>  mapping(fe_degree);
      const QGauss<dim>    quad_3d(nq);
      const FiniteElement<dim> &fe = dof_handler_ref.get_fe();
      FEValues<dim>        fe_values(
        mapping, fe, quad_3d, update_gradients | update_JxW_values);

      const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
      std::vector<types::global_dof_index> cell_dofs(dofs_per_cell);
      std::vector<Number>                  u_loc(dofs_per_cell);

      for (const auto &cell : dof_handler_ref.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell->get_dof_indices(cell_dofs);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              u_loc[i] = constraints_ref.is_constrained(cell_dofs[i]) ?
                           Number(0) :
                           src_host(cell_dofs[i]);

            for (unsigned int q = 0; q < quad_3d.size(); ++q)
              {
                Tensor<1, dim, Number> grad_u_q;
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  for (int d = 0; d < dim; ++d)
                    grad_u_q[d] +=
                      u_loc[j] *
                      static_cast<Number>(fe_values.shape_grad(j, q)[d]);

                const Number jxw = static_cast<Number>(fe_values.JxW(q));
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  if (!constraints_ref.is_constrained(cell_dofs[i]))
                    {
                      Number accum = Number(0);
                      for (int d = 0; d < dim; ++d)
                        accum +=
                          grad_u_q[d] *
                          static_cast<Number>(fe_values.shape_grad(i, q)[d]);
                      dst_host(cell_dofs[i]) += accum * jxw;
                    }
              }
          }

      dst_host.compress(VectorOperation::add);

      for (const auto &line : constraints_ref.get_lines())
        if (host_partitioner->in_local_range(line.index))
          {
            const unsigned int loc =
              host_partitioner->global_to_local(line.index);
            dst_host.local_element(loc) = src_host.local_element(loc);
          }

      copy_owned_host_to_device(dst_host, dst);
    }

  private:
    const DoFHandler<dim>                              &dof_handler_ref;
    const AffineConstraints<Number>                    &constraints_ref;
    bool                                                use_host_mpi_fallback;
    std::shared_ptr<const Utilities::MPI::Partitioner>  host_partitioner;
    Portable::MatrixFree<dim, Number>                   matrix_free;
  };

  template <int dim, int fe_degree, typename Number>
  void
  run_deformed_operator_test(MPI_Comm           comm,
                             ConditionalOStream &pcout,
                             const double       rel_tol,
                             const unsigned int n_refine   = 2u,
                             const unsigned int sub_x      = 1u)
  {
    constexpr int nq = fe_degree + 2;

    parallel::distributed::Triangulation<dim> tria(comm);
    const std::vector<unsigned int>           reps = {sub_x, 1u, 1u};
    GridGenerator::subdivided_hyper_rectangle(
      tria, reps, Point<dim>(0.0, 0.0, 0.0), Point<dim>(1.0, 1.0, 1.0));
    GridTools::transform(&CurvedShearedManifold<dim>::map_forward, tria);

    CurvedShearedManifold<dim> manifold;
    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, manifold);
    tria.refine_global(n_refine);

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    const auto shared_part = std::make_shared<Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, comm);

    AffineConstraints<Number> constraints;
    constraints.reinit(locally_owned, locally_relevant);
    VectorTools::interpolate_boundary_values(
      MappingQ<dim>(fe_degree),
      dof_handler,
      0,
      Functions::ZeroFunction<dim, Number>(),
      constraints);
    constraints.close();

    DealIIReferenceLaplaceOperator<dim, fe_degree, nq, Number> ref_op(
      dof_handler, constraints);
    PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, Number> opt2_no_ovl(
      dof_handler, constraints, /*overlap=*/false);
    PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, Number> opt2_ovl(
      dof_handler, constraints, /*overlap=*/true, shared_part);

    AssertThrow(opt2_no_ovl.m() == dof_handler.n_dofs() &&
                  opt2_ovl.n() == dof_handler.n_dofs(),
                ExcMessage("Operator dimensions m()/n() mismatch!"));

    using Vec = LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>;
    Vec src, dst_ref, dst_no_ovl, dst_ovl, dst_tvmult, dst_vmult_add,
      dst_async, base_vec;
    ref_op.initialize_dof_vector(src);
    ref_op.initialize_dof_vector(dst_ref);
    opt2_no_ovl.initialize_dof_vector(dst_no_ovl);
    opt2_ovl.initialize_dof_vector(dst_ovl);
    opt2_ovl.initialize_dof_vector(dst_tvmult);
    opt2_ovl.initialize_dof_vector(dst_vmult_add);
    opt2_ovl.initialize_dof_vector(dst_async);
    opt2_ovl.initialize_dof_vector(base_vec);

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> src_host_init(
      shared_part);
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
      base_host_init(shared_part);
    for (const auto idx : locally_owned)
      {
        src_host_init(idx) = static_cast<Number>(
          std::sin(0.37 * static_cast<double>(idx + 1)) +
          0.5 * std::cos(0.13 * static_cast<double>(idx + 3)));
        base_host_init(idx) =
          static_cast<Number>(0.25 * std::cos(0.19 * static_cast<double>(idx + 5)));
      }
    copy_owned_host_to_device(src_host_init, src);
    copy_owned_host_to_device(base_host_init, base_vec);
    dst_vmult_add = base_vec;
    Kokkos::fence();

    ref_op.vmult(dst_ref, src);
    opt2_no_ovl.vmult(dst_no_ovl, src);
    opt2_ovl.vmult(dst_ovl, src);
    opt2_ovl.Tvmult(dst_tvmult, src);
    opt2_ovl.vmult_add(dst_vmult_add, src);

    const auto &comp_exec = opt2_ovl.get_compute_exec();
    opt2_ovl.vmult_async(comp_exec, dst_async, src);
    comp_exec.fence();

    //Also test vmult_dummy modes so ghost-only and compute-only never corrupt state
    Vec dummy_dst;
    opt2_ovl.initialize_dof_vector(dummy_dst);
    opt2_ovl.vmult_dummy(dummy_dst, src, true, false);
    opt2_ovl.vmult_dummy(dummy_dst, src, false, true);
    opt2_ovl.vmult_dummy(dummy_dst, src, true, true);

    const double ref_norm = dst_ref.l2_norm();
    AssertThrow(ref_norm > 0.0, ExcMessage("Reference operator norm is zero."));

    Vec diff(dst_ref);
    diff = dst_no_ovl;
    diff -= dst_ref;
    const double err_no_ovl = diff.l2_norm() / ref_norm;

    diff = dst_ovl;
    diff -= dst_ref;
    const double err_ovl = diff.l2_norm() / ref_norm;

    diff = dst_tvmult;
    diff -= dst_ref;
    const double err_tvmult = diff.l2_norm() / ref_norm;

    diff = dst_async;
    diff -= dst_ref;
    const double err_async = diff.l2_norm() / ref_norm;

    diff = dummy_dst;
    diff -= dst_ref;
    const double err_dummy_both = diff.l2_norm() / ref_norm;

    Vec expected_add(dst_ref);
    expected_add += base_vec;
    diff = dst_vmult_add;
    diff -= expected_add;
    const double err_add = diff.l2_norm() / expected_add.l2_norm();

    // Verify constant RHS assembly
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Host> rhs_host(
      locally_owned, locally_relevant, comm);
    const MappingQ<dim> mapping_q(fe_degree);
    const QGauss<dim>   quad_3d(nq);
    FEValues<dim>       fe_values(
      mapping_q, fe, quad_3d, update_values | update_JxW_values);
    Vector<Number>                       cell_rhs(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> cell_dofs(fe.n_dofs_per_cell());
    constexpr Number                     rhs_const = static_cast<Number>(1.7);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = Number(0);
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < quad_3d.size(); ++q)
            {
              const Number jxw =
                rhs_const * static_cast<Number>(fe_values.JxW(q));
              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
                cell_rhs(i) +=
                  static_cast<Number>(fe_values.shape_value(i, q)) * jxw;
            }
          cell->get_dof_indices(cell_dofs);
          constraints.distribute_local_to_global(cell_rhs, cell_dofs, rhs_host);
        }
    rhs_host.compress(VectorOperation::add);

    Vec rhs_ref_dev, rhs_no_ovl, rhs_ovl;
    opt2_no_ovl.initialize_dof_vector(rhs_ref_dev);
    opt2_no_ovl.initialize_dof_vector(rhs_no_ovl);
    opt2_ovl.initialize_dof_vector(rhs_ovl);
    copy_owned_host_to_device(rhs_host, rhs_ref_dev);

    opt2_no_ovl.assemble_constant_rhs(rhs_no_ovl, rhs_const);
    opt2_ovl.assemble_constant_rhs(rhs_ovl, rhs_const);

    const double rhs_ref_norm = rhs_ref_dev.l2_norm();
    diff                      = rhs_no_ovl;
    diff -= rhs_ref_dev;
    const double rhs_err_no_ovl = diff.l2_norm() / rhs_ref_norm;

    diff = rhs_ovl;
    diff -= rhs_ref_dev;
    const double rhs_err_ovl = diff.l2_norm() / rhs_ref_norm;

    pcout << "  [Opt2 Deformed Q" << fe_degree << ", "
          << (std::is_same_v<Number, double> ? "fp64" : "fp32")
          << ", cells=" << std::setw(4) << tria.n_global_active_cells()
          << ", dofs=" << std::setw(5) << dof_handler.n_dofs()
          << "] err(no_ovl)=" << std::scientific << std::setprecision(3)
          << err_no_ovl << " | err(ovl)=" << err_ovl
          << " | err(add)=" << err_add
          << " | rhs_err=" << std::max(rhs_err_no_ovl, rhs_err_ovl)
          << std::endl;

    AssertThrow(err_no_ovl < rel_tol,
                ExcMessage("Opt2 vmult (overlap=false) failed tolerance!"));
    AssertThrow(err_ovl < rel_tol,
                ExcMessage("Opt2 vmult (overlap=true) failed tolerance!"));
    AssertThrow(err_tvmult < rel_tol,
                ExcMessage("Opt2 Tvmult failed tolerance!"));
    AssertThrow(err_async < rel_tol,
                ExcMessage("Opt2 vmult_async failed tolerance!"));
    AssertThrow(err_dummy_both < rel_tol,
                ExcMessage("Opt2 vmult_dummy(true,true) failed tolerance!"));
    AssertThrow(err_add < rel_tol,
                ExcMessage("Opt2 vmult_add failed tolerance!"));
    AssertThrow(rhs_err_no_ovl < rel_tol && rhs_err_ovl < rel_tol,
                ExcMessage("Opt2 assemble_constant_rhs failed tolerance!"));
  }

  template <int dim, int fe_degree>
  void
  run_cg_modes_equivalence_test(MPI_Comm comm, ConditionalOStream &pcout)
  {
    constexpr int nq = fe_degree + 2;

    parallel::distributed::Triangulation<dim> tria(comm);
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    GridTools::transform(&CurvedShearedManifold<dim>::map_forward, tria);
    CurvedShearedManifold<dim> manifold;
    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, manifold);
    tria.refine_global(2);

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    const auto shared_part = std::make_shared<Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, comm);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned, locally_relevant);
    VectorTools::interpolate_boundary_values(MappingQ<dim>(fe_degree),
                                             dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, double> op(
      dof_handler, constraints, /*overlap=*/true, shared_part);

    using Vec = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
    Vec rhs, x_dealii, x_fused_exact, x_single_ar, x_nonzero_dealii,
      x_nonzero_fused;
    op.initialize_dof_vector(rhs);
    op.initialize_dof_vector(x_dealii);
    op.initialize_dof_vector(x_fused_exact);
    op.initialize_dof_vector(x_single_ar);
    op.initialize_dof_vector(x_nonzero_dealii);
    op.initialize_dof_vector(x_nonzero_fused);

    op.assemble_constant_rhs(rhs, 1.3);

    GrowingVectorMemory<Vec> mem;

    // 1. Stock dealii::SolverCG directly on LaplaceOperatorOpt2
    ReductionControl ctrl_dealii(5000, 1e-16, 1e-12);
    {
      SolverCG<Vec> cg_ref(ctrl_dealii, mem);
      x_dealii = 0.0;
      cg_ref.solve(op, x_dealii, rhs, PreconditionIdentity());
    }

    // 2. SolverCGOpt2 (FusedExact default)
    ReductionControl ctrl_fused(5000, 1e-16, 1e-12);
    {
      PortableOpt2::SolverCGOpt2<Vec> cg_fused(
        ctrl_fused,
        mem,
        typename PortableOpt2::SolverCGOpt2<Vec>::AdditionalData(
          PortableOpt2::CGMode::FusedExact));
      x_fused_exact = 0.0;
      cg_fused.solve(op, x_fused_exact, rhs, PreconditionIdentity());
    }

    // 3. SolverCGOpt2 (FusedSingleAllreduce)
    ReductionControl ctrl_single_ar(5000, 1e-16, 1e-12);
    {
      PortableOpt2::SolverCGOpt2<Vec> cg_single(
        ctrl_single_ar,
        mem,
        typename PortableOpt2::SolverCGOpt2<Vec>::AdditionalData(
          PortableOpt2::CGMode::FusedSingleAllreduce));
      x_single_ar = 0.0;
      cg_single.solve(op, x_single_ar, rhs, PreconditionIdentity());
    }

    // 4. Non-zero initial guess x_0 != 0 test
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> x_init_host(
      shared_part);
    for (const auto idx : locally_owned)
      x_init_host(idx) =
        constraints.is_constrained(idx) ?
          0.0 :
          0.05 * std::sin(0.21 * static_cast<double>(idx + 1));
    copy_owned_host_to_device(x_init_host, x_nonzero_dealii);
    copy_owned_host_to_device(x_init_host, x_nonzero_fused);

    ReductionControl ctrl_nz_dealii(5000, 1e-16, 1e-12);
    {
      SolverCG<Vec> cg_ref(ctrl_nz_dealii, mem);
      cg_ref.solve(op, x_nonzero_dealii, rhs, PreconditionIdentity());
    }
    ReductionControl ctrl_nz_fused(5000, 1e-16, 1e-12);
    {
      PortableOpt2::SolverCGOpt2<Vec> cg_fused(ctrl_nz_fused, mem);
      cg_fused.solve(op, x_nonzero_fused, rhs, PreconditionIdentity());
    }

    const double ref_norm = x_dealii.l2_norm();
    Vec          diff(x_dealii);

    diff = x_fused_exact;
    diff -= x_dealii;
    const double err_fused = diff.l2_norm() / ref_norm;

    diff = x_single_ar;
    diff -= x_dealii;
    const double err_single_ar = diff.l2_norm() / ref_norm;

    diff = x_nonzero_fused;
    diff -= x_nonzero_dealii;
    const double err_nonzero = diff.l2_norm() / x_nonzero_dealii.l2_norm();

    pcout << "  [Opt2 CG Modes Q" << fe_degree
          << ", its(dealii/fused/1-AR)=" << ctrl_dealii.last_step() << "/"
          << ctrl_fused.last_step() << "/" << ctrl_single_ar.last_step()
          << "] err(fused)=" << std::scientific << std::setprecision(3)
          << err_fused << " | err(1-AR)=" << err_single_ar
          << " | err(x0!=0)=" << err_nonzero << std::endl;

    AssertThrow(ctrl_fused.last_step() == ctrl_dealii.last_step(),
                ExcMessage(
                  "FusedExact CG iteration count differed from dealii::SolverCG!"));
    AssertThrow(ctrl_nz_fused.last_step() == ctrl_nz_dealii.last_step(),
                ExcMessage(
                  "FusedExact CG (x0!=0) iteration count differed from dealii::SolverCG!"));
    AssertThrow(err_fused < 1e-12,
                ExcMessage("FusedExact CG solution diverged from dealii::SolverCG!"));
    AssertThrow(err_single_ar < 1e-10,
                ExcMessage("FusedSingleAllreduce CG solution diverged from dealii::SolverCG!"));
    AssertThrow(err_nonzero < 1e-11,
                ExcMessage("FusedExact CG (x0!=0) diverged from dealii::SolverCG!"));
  }

  template <int dim>
  class ManufacturedSolution : public Function<dim>
  {
  public:
    explicit ManufacturedSolution(const Point<dim> &extents)
      : Function<dim>(1)
      , L(extents)
    {}

    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return std::sin(numbers::PI * p[0] / L[0]) *
             std::sin(numbers::PI * p[1] / L[1]) *
             std::sin(numbers::PI * p[2] / L[2]);
    }

  private:
    Point<dim> L;
  };

  template <int dim>
  class ManufacturedRHS : public Function<dim>
  {
  public:
    explicit ManufacturedRHS(const Point<dim> &extents)
      : Function<dim>(1)
      , L(extents)
      , factor(numbers::PI * numbers::PI *
               (1.0 / (extents[0] * extents[0]) +
                1.0 / (extents[1] * extents[1]) +
                1.0 / (extents[2] * extents[2])))
    {}

    double
    value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return factor * std::sin(numbers::PI * p[0] / L[0]) *
             std::sin(numbers::PI * p[1] / L[1]) *
             std::sin(numbers::PI * p[2] / L[2]);
    }

  private:
    Point<dim> L;
    double     factor;
  };

  template <int dim, int fe_degree>
  double
  solve_manufactured_level(MPI_Comm           comm,
                           const Point<dim>  &extents,
                           const unsigned int n_refine)
  {
    constexpr int nq = fe_degree + 2;

    parallel::distributed::Triangulation<dim> tria(comm);
    GridGenerator::hyper_rectangle(tria, Point<dim>(), extents);
    tria.refine_global(n_refine);

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    const auto shared_part = std::make_shared<Utilities::MPI::Partitioner>(
      locally_owned, locally_relevant, comm);

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned, locally_relevant);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    const bool use_overlap = (Utilities::MPI::n_mpi_processes(comm) > 1);
    PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, double> op(
      dof_handler, constraints, use_overlap, shared_part);

    using Vec = LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;
    Vec sol_dev, rhs_dev;
    op.initialize_dof_vector(sol_dev);
    op.initialize_dof_vector(rhs_dev);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> rhs_host(
      locally_owned, locally_relevant, comm);
    const QGauss<dim>  quad(nq);
    FEValues<dim>      fe_values(fe,
                            quad,
                            update_values | update_quadrature_points |
                              update_JxW_values);
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const ManufacturedRHS<dim>           rhs_func(extents);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_rhs = 0.0;
          fe_values.reinit(cell);
          for (unsigned int q = 0; q < quad.size(); ++q)
            {
              const double f_val =
                rhs_func.value(fe_values.quadrature_point(q)) *
                fe_values.JxW(q);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                cell_rhs(i) += fe_values.shape_value(i, q) * f_val;
            }
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 rhs_host);
        }
    rhs_host.compress(VectorOperation::add);
    copy_owned_host_to_device(rhs_host, rhs_dev);

    ReductionControl                    solver_control(5000, 1e-16, 1e-12);
    PortableOpt2::SolverCGOpt2<Vec>     cg(solver_control);
    sol_dev = 0.0;
    cg.solve(op, sol_dev, rhs_dev, PreconditionIdentity());
    Kokkos::fence();

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host> sol_host;
    if (op.get_vector_partitioner())
      sol_host.reinit(op.get_vector_partitioner());
    else
      sol_host.reinit(dof_handler.n_dofs());
    Kokkos::deep_copy(
      Kokkos::View<double *, Kokkos::HostSpace>(sol_host.get_values(),
                                                sol_host.locally_owned_size()),
      Kokkos::View<const double *, MemorySpace::Default::kokkos_space>(
        sol_dev.get_values(), sol_dev.locally_owned_size()));
    sol_host.update_ghost_values();

    Vector<float>                   diff_per_cell(tria.n_active_cells());
    const ManufacturedSolution<dim> exact_sol(extents);
    VectorTools::integrate_difference(dof_handler,
                                      sol_host,
                                      exact_sol,
                                      diff_per_cell,
                                      QGauss<dim>(nq + 1),
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(tria,
                                             diff_per_cell,
                                             VectorTools::L2_norm);
  }

  template <int dim, int fe_degree>
  void
  run_convergence_test(MPI_Comm comm, ConditionalOStream &pcout)
  {
    const Point<dim> extents(1.7, 1.1, 0.8);
    const double e1 = solve_manufactured_level<dim, fe_degree>(comm, extents, 2);
    const double e2 = solve_manufactured_level<dim, fe_degree>(comm, extents, 3);
    const double rate = std::log2(e1 / e2);

    pcout << "  [Opt2 Convergence Q" << fe_degree
          << "] L2(ref=2)=" << std::scientific << std::setprecision(3) << e1
          << " -> L2(ref=3)=" << e2 << " | rate=" << std::fixed
          << std::setprecision(2) << rate << " (expected ~" << (fe_degree + 1)
          << ".00)" << std::endl;

    AssertThrow(rate > static_cast<double>(fe_degree) + 0.75,
                ExcMessage("Manufactured solution convergence rate too low!"));
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
      MPI_Comm                         comm = MPI_COMM_WORLD;
      ConditionalOStream               pcout(
        std::cout, Utilities::MPI::this_mpi_process(comm) == 0);

      pcout << "=== CEED BP3 Opt2 Verification Suite (MPI ranks = "
            << Utilities::MPI::n_mpi_processes(comm)
            << ", GPU-aware MPI = "
            << (PortableOpt2::query_device_aware_mpi() ? "true" :
                                                         "false (pinned-DMA staging)")
            << ") ===" << std::endl;

      // 1. Deformed, anisotropic, sheared & curved MappingQ(p) operator tests
      //    across ALL polynomial orders p = 1..8 (fp64) and odd-nq fp32
      run_deformed_operator_test<3, 1, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 2, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 3, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 4, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 5, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 6, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 7, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 8, double>(comm, pcout, 1e-12);

      run_deformed_operator_test<3, 1, float>(comm, pcout, 5e-5);
      run_deformed_operator_test<3, 3, float>(comm, pcout, 5e-5);

      // 2. Multi-tile (> 4096 cells/rank) double-buffered pinned DMA setup test
      run_deformed_operator_test<3, 1, double>(comm, pcout, 1e-12, /*n_refine=*/4u, /*sub_x=*/2u);

      // 3. SolverCGOpt2 (FusedExact, FusedSingleAllreduce, x0 != 0) vs dealii::SolverCG
      run_cg_modes_equivalence_test<3, 2>(comm, pcout);
      run_cg_modes_equivalence_test<3, 4>(comm, pcout);

      // 4. Manufactured-solution PDE convergence rates
      run_convergence_test<3, 2>(comm, pcout);
      run_convergence_test<3, 3>(comm, pcout);

      pcout << "=== ALL OPT2 VERIFICATION TESTS PASSED ===" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << "Opt2 Verification FAILED: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
