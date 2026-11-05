#ifndef portable_laplace_operator_opt2_h
#define portable_laplace_operator_opt2_h

#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/shape_info.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bk3_kokkos_kernel_opt.h"
#include "portable_halo_exchanger_opt2.h"

namespace PortableOpt2
{
  using namespace dealii;

  namespace internal
  {
    template <int dim, int nm, int nq, typename number>
    struct ConstantRhsKernelFunctorOpt2
    {
      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using MemberType = typename TeamPolicy::member_type;
      using SharedView = BK3Opt::Parallel::SharedView<number>;

      BK3Opt::Parallel::ConstDeviceView<number> d_shape_values;
      BK3Opt::Parallel::ConstDeviceView<number> d_weights_1d;
      BK3Opt::Parallel::ConstDeviceView<number> d_G;
      BK3Opt::Parallel::DeviceView<number>      d_out;
      BK3Opt::Parallel::ConstDoFIndicesView     dof_indices;
      number                                    rhs_value;
      unsigned int                              n_cells;
      unsigned int                              nelmtPerBatch;
      unsigned int                              total_batches;
      std::size_t                               shmem_size;
      bool                                      use_coloring;

      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        return shmem_size;
      }

      KOKKOS_INLINE_FUNCTION void
      operator()(const MemberType &team_member) const
      {
        constexpr unsigned int nq_total = Utilities::pow(nq, dim);

        SharedView s_shape_values(team_member.team_shmem(), nm * nq);
        SharedView s_weights(team_member.team_shmem(), nq);
        SharedView s_wsp0(team_member.team_shmem(), nelmtPerBatch * nq_total);
        SharedView s_wsp1(team_member.team_shmem(), nelmtPerBatch * nq_total);

        const unsigned int threadIdx = team_member.team_rank();
        const unsigned int blockSize = team_member.team_size();

        for (unsigned int tid = threadIdx;
             tid < static_cast<unsigned int>(nm * nq);
             tid += blockSize)
          s_shape_values[tid] = d_shape_values[tid];

        for (unsigned int tid = threadIdx; tid < static_cast<unsigned int>(nq);
             tid += blockSize)
          s_weights[tid] = d_weights_1d[tid];

        team_member.team_barrier();

        unsigned int eb = team_member.league_rank();
        while (eb < total_batches)
          {
            const unsigned int c_nelmtPerBatch =
              (eb * nelmtPerBatch + nelmtPerBatch > n_cells) ?
                (n_cells - eb * nelmtPerBatch) :
                nelmtPerBatch;

            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nq * nq);
                 tid += blockSize)
              {
                const unsigned int e       = tid / (nq * nq);
                const unsigned int q       = (tid % (nq * nq)) / nq;
                const unsigned int p       = tid % nq;
                const unsigned int cell_id = eb * nelmtPerBatch + e;
                const std::size_t  g_base =
                  static_cast<std::size_t>(cell_id) * 6 * nq_total;

                const number w_qp = s_weights[q] * s_weights[p];
                number       r_rhs[nq];

                for (unsigned int r = 0; r < static_cast<unsigned int>(nq); ++r)
                  {
                    const std::size_t q_pt =
                      static_cast<std::size_t>(p * nq * nq + q * nq + r);

                    const number G00 = d_G[g_base + 0 * nq_total + q_pt];
                    const number G01 = d_G[g_base + 1 * nq_total + q_pt];
                    const number G02 = d_G[g_base + 2 * nq_total + q_pt];
                    const number G11 = d_G[g_base + 3 * nq_total + q_pt];
                    const number G12 = d_G[g_base + 4 * nq_total + q_pt];
                    const number G22 = d_G[g_base + 5 * nq_total + q_pt];

                    const number detG = G00 * (G11 * G22 - G12 * G12) -
                                        G01 * (G01 * G22 - G12 * G02) +
                                        G02 * (G01 * G12 - G11 * G02);
                    const number w3d = s_weights[r] * w_qp;
                    r_rhs[r]         = rhs_value * (detG / (w3d * w3d));
                  }

                for (unsigned int k = 0; k < static_cast<unsigned int>(nm); ++k)
                  {
                    number tmp = 0;
                    for (unsigned int r = 0; r < static_cast<unsigned int>(nq);
                         ++r)
                      tmp += s_shape_values[k * nq + r] * r_rhs[r];

                    s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
                  }
              }
            team_member.team_barrier();

            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nq);
                 tid += blockSize)
              {
                const unsigned int e = tid / (nm * nq);
                const unsigned int k = (tid % (nm * nq)) / nq;
                const unsigned int p = tid % nq;

                number r_q[nq];
                for (unsigned int q = 0; q < static_cast<unsigned int>(nq); ++q)
                  r_q[q] = s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p];

                for (unsigned int j = 0; j < static_cast<unsigned int>(nm); ++j)
                  {
                    number tmp = 0;
                    for (unsigned int q = 0; q < static_cast<unsigned int>(nq);
                         ++q)
                      tmp += s_shape_values[j * nq + q] * r_q[q];

                    s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
                  }
              }
            team_member.team_barrier();

            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nm);
                 tid += blockSize)
              {
                const unsigned int e       = tid / (nm * nm);
                const unsigned int j       = (tid % (nm * nm)) / nm;
                const unsigned int k       = tid % nm;
                const unsigned int cell_id = eb * nelmtPerBatch + e;

                number r_p[nq];
                for (unsigned int p = 0; p < static_cast<unsigned int>(nq); ++p)
                  r_p[p] = s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p];

                for (unsigned int i = 0; i < static_cast<unsigned int>(nm); ++i)
                  {
                    const unsigned int local_idx = i * nm * nm + j * nm + k;
                    const unsigned int dof_index =
                      dof_indices(local_idx, cell_id);
                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        number tmp = 0;
                        for (unsigned int p = 0;
                             p < static_cast<unsigned int>(nq);
                             ++p)
                          tmp += s_shape_values[i * nq + p] * r_p[p];

                        if (use_coloring)
                          d_out[dof_index] += tmp;
                        else
                          Kokkos::atomic_add(&d_out[dof_index], tmp);
                      }
                  }
              }
            team_member.team_barrier();

            eb += team_member.league_size();
          }
      }
    };

    template <int dim,
              int nq,
              typename number,
              typename InvJacView,
              typename JxWView,
              typename GView>
    struct FillGTensorTileCoalescedFunctor
    {
      static constexpr unsigned int n_q_points = Utilities::pow(nq, dim);
      static constexpr int          symmetric_tensor_dim = (dim * (dim + 1)) / 2;

      unsigned int cell_offset;
      InvJacView   inv_jacobian_tile;
      JxWView      JxW_tile;
      GView        G;

      KOKKOS_INLINE_FUNCTION void
      operator()(const std::size_t idx) const
      {
        const unsigned int t =
          static_cast<unsigned int>(idx / static_cast<std::size_t>(n_q_points));
        const unsigned int q_point =
          static_cast<unsigned int>(idx % static_cast<std::size_t>(n_q_points));
        const unsigned int cell_id = cell_offset + t;
        const std::size_t  g_cell_base =
          static_cast<std::size_t>(cell_id) * symmetric_tensor_dim * n_q_points;

        const number jxw_val = JxW_tile(t, q_point);

        int c = 0;
        for (int d1 = 0; d1 < dim; ++d1)
          for (int d2 = d1; d2 < dim; ++d2)
            {
              number sum = 0;
              for (int k = 0; k < dim; ++k)
                sum += inv_jacobian_tile(t, q_point, d1, k) *
                       inv_jacobian_tile(t, q_point, d2, k);
              G[g_cell_base + static_cast<std::size_t>(c) * n_q_points +
                q_point] = jxw_val * sum;
              ++c;
            }
      }
    };

    template <typename number, bool add_mode>
    struct ApplyConstrainedValuesFunctor
    {
      Kokkos::View<const unsigned int *, DeviceMemSpace> constr_dofs;
      const number                                      *src_ptr;
      number                                            *dst_ptr;

      KOKKOS_INLINE_FUNCTION void
      operator()(const unsigned int i) const
      {
        const unsigned int dof = constr_dofs(i);
        if constexpr (add_mode)
          dst_ptr[dof] += src_ptr[dof];
        else
          dst_ptr[dof] = src_ptr[dof];
      }
    };

  } // namespace internal

  template <int dim, int fe_degree, int nq, typename number>
  class LaplaceOperatorOpt2
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;
    using ExecutionSpace = internal::ExecutionSpace;

    LaplaceOperatorOpt2(
      const DoFHandler<dim>           &dof_handler,
      const AffineConstraints<number> &constraints,
      const bool                       overlap_communication_computation,
      const std::shared_ptr<const Utilities::MPI::Partitioner>
        &external_partitioner = nullptr);

    void
    vmult(VectorType &dst, const VectorType &src) const;

    void
    vmult_async(const ExecutionSpace &caller_exec,
                VectorType           &dst,
                const VectorType     &src) const;

    void
    vmult_add(VectorType &dst, const VectorType &src) const;

    void
    vmult_dummy(VectorType       &dst,
                const VectorType &src,
                const bool        ghost_exchange_on,
                const bool        computation_on) const;

    void
    Tvmult(VectorType &dst, const VectorType &src) const;

    void
    assemble_constant_rhs(VectorType  &dst,
                          const number rhs_value = number(1)) const;

    void
    initialize_dof_vector(VectorType &vec) const;

    types::global_dof_index
    m() const;

    types::global_dof_index
    n() const;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const;

    const ExecutionSpace &
    get_compute_exec() const
    {
      return stream_ctx.get_compute_exec();
    }

    const ExecutionSpace &
    get_comm_exec() const
    {
      return stream_ctx.get_comm_exec();
    }

    BK3Opt::Parallel::ConstDeviceView<number>
    get_G_tensor(const unsigned int color = 0) const;

    std::size_t
    memory_consumption() const;

  private:
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points = Utilities::pow(nq, dim);

    unsigned int
    compute_optimal_num_blocks(const unsigned int n_cells) const;

    void
    setup_operator(const DoFHandler<dim>           &dof_handler,
                   const AffineConstraints<number> &constraints,
                   const std::shared_ptr<const Utilities::MPI::Partitioner>
                     &external_partitioner);

    void
    apply_constrained_values(const VectorType     &src,
                             VectorType           &dst,
                             const ExecutionSpace &exec,
                             const bool            add_mode = false) const;

    void
    launch_color_on_stream(const ExecutionSpace                  &exec,
                           const unsigned int                     color,
                           const Portable::DeviceVector<number>  &src_device,
                           Portable::DeviceVector<number>        &dst_device) const;

    internal::StreamContextOpt2   stream_ctx;
    internal::DeviceStreamSyncEvent caller_ready_event;
    internal::DeviceStreamSyncEvent dst_zeroed_event;
    internal::DeviceStreamSyncEvent comm_done_event;
    HaloExchangerOpt2<number>       halo_exchanger;

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    types::global_dof_index                            total_dofs         = 0;
    unsigned int                                       locally_owned_size = 0;
    unsigned int                                       ghost_size         = 0;
    bool has_ghost_exchange                = false;
    bool overlap_communication_computation = false;

    unsigned int              n_colors = 0;
    std::vector<unsigned int> n_cells_per_color;
    std::vector<unsigned int> optimal_num_blocks_per_color;
    std::vector<bool>         use_coloring_per_color;

    Kokkos::View<number *, internal::DeviceMemSpace> shape_values;
    Kokkos::View<number *, internal::DeviceMemSpace> co_shape_gradients;
    Kokkos::View<number *, internal::DeviceMemSpace> quad_weights_1d;

    Kokkos::View<unsigned int *, internal::DeviceMemSpace>
      owned_constrained_dofs;

    std::vector<Kokkos::View<unsigned int **, internal::DeviceMemSpace>>
      dirichlet_boundary_dofs_masks;

    std::vector<Kokkos::View<number *, internal::DeviceMemSpace>> G_tensors;
  };

  template <int dim, int fe_degree, int nq, typename number>
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::LaplaceOperatorOpt2(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const bool                       overlap_communication_computation,
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &external_partitioner)
    : total_dofs(dof_handler.n_dofs())
    , overlap_communication_computation(overlap_communication_computation)
  {
    for (const auto &line : constraints.get_lines())
      {
        AssertThrow(
          line.entries.empty() && line.inhomogeneity == number(0),
          ExcMessage(
            "LaplaceOperatorOpt2 only supports homogeneous Dirichlet constraints; "
            "hanging-node or inhomogeneous constraints were detected."));
      }

    setup_operator(dof_handler, constraints, external_partitioner);
  }

  template <int dim, int fe_degree, int nq, typename number>
  unsigned int
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::compute_optimal_num_blocks(
    const unsigned int n_cells) const
  {
    if (n_cells == 0u)
      return 0u;

    constexpr bool is_serial =
      std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
    if constexpr (is_serial)
      return 1u;

    // Match BK3Opt::Parallel::KokkosKernelAsync's exact batch-size calculation
    constexpr unsigned int shmem_per_block = 10800u;
    unsigned int           nelmt_per_batch =
      shmem_per_block / (4u * n_q_points * static_cast<unsigned int>(sizeof(number)));
    if (nelmt_per_batch == 0u)
      nelmt_per_batch = 1u;

    const unsigned int total_batches =
      (n_cells + nelmt_per_batch - 1u) / nelmt_per_batch;
    const unsigned int target_wave_blocks =
      std::max(1u, 2u * stream_ctx.get_n_compute_units());

    // Below full-GPU saturation: launch 1 block per batch so SMs are never left
    // idle while active blocks loop sequentially. Above saturation: transition
    // smoothly to ~2 batches/block to amortize shared-memory table prologues.
    if (total_batches <= target_wave_blocks)
      return total_batches;

    return std::max(target_wave_blocks, (total_batches + 1u) / 2u);
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::setup_operator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &external_partitioner)
  {
    AssertDimension(dim, 3);
    AssertDimension(dof_handler.get_fe().n_dofs_per_cell(), n_local_dofs);

    const ExecutionSpace &comp_exec = stream_ctx.get_compute_exec();
    const ExecutionSpace &comm_exec = stream_ctx.get_comm_exec();

    // 1. Reuse caller-supplied MPI Partitioner when provided, avoiding
    //    duplicate DoFTools::extract_locally_relevant_dofs + MPI rendezvous
    const auto *parallel_tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &dof_handler.get_triangulation());
    if (external_partitioner != nullptr)
      {
        this->partitioner = external_partitioner;
      }
    else if (parallel_tria != nullptr)
      {
        const IndexSet locally_relevant_dofs =
          DoFTools::extract_locally_relevant_dofs(dof_handler);
        this->partitioner = std::make_shared<Utilities::MPI::Partitioner>(
          dof_handler.locally_owned_dofs(),
          locally_relevant_dofs,
          parallel_tria->get_mpi_communicator());
      }

    this->locally_owned_size =
      partitioner ? partitioner->locally_owned_size() :
                    static_cast<unsigned int>(total_dofs);
    this->ghost_size = partitioner ? partitioner->n_ghost_indices() : 0u;
    this->has_ghost_exchange =
      partitioner && (partitioner->n_mpi_processes() > 1u) &&
      (partitioner->n_ghost_indices() > 0u ||
       partitioner->n_import_indices() > 0u);

    if (partitioner)
      halo_exchanger.reinit(partitioner, comp_exec);

    const std::size_t n_mpi_local_dofs =
      static_cast<std::size_t>(locally_owned_size) + ghost_size;
    AssertThrow(
      n_mpi_local_dofs <
        static_cast<std::size_t>(numbers::invalid_unsigned_int),
      ExcMessage("Local DoF index exceeds 32-bit unsigned int range."));

    // 2. Upload 1D shape values, collocation derivatives & quadrature weights
    //    from pinned host memory asynchronously
    const QGauss<1> quadrature_1d(nq);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<number> shape_info(
      quadrature_1d, dof_handler.get_fe());

    constexpr unsigned int n_dofs_1d    = fe_degree + 1;
    const unsigned int     n_shape_vals = n_dofs_1d * nq;

    this->shape_values = Kokkos::View<number *, internal::DeviceMemSpace>(
      Kokkos::view_alloc("opt2_shape_values", Kokkos::WithoutInitializing),
      n_shape_vals);
    this->co_shape_gradients = Kokkos::View<number *, internal::DeviceMemSpace>(
      Kokkos::view_alloc("opt2_co_shape_gradients",
                         Kokkos::WithoutInitializing),
      nq * nq);
    this->quad_weights_1d = Kokkos::View<number *, internal::DeviceMemSpace>(
      Kokkos::view_alloc("opt2_quad_weights_1d", Kokkos::WithoutInitializing),
      nq);

    Kokkos::View<number *, internal::PinnedHostMemSpace> sv_pinned(
      Kokkos::view_alloc("opt2_sv_pinned", Kokkos::WithoutInitializing),
      n_shape_vals);
    Kokkos::View<number *, internal::PinnedHostMemSpace> cg_pinned(
      Kokkos::view_alloc("opt2_cg_pinned", Kokkos::WithoutInitializing),
      nq * nq);
    Kokkos::View<number *, internal::PinnedHostMemSpace> qw_pinned(
      Kokkos::view_alloc("opt2_qw_pinned", Kokkos::WithoutInitializing), nq);

    for (unsigned int i = 0; i < n_shape_vals; ++i)
      sv_pinned(i) = shape_info.data.front().shape_values[i];
    for (int i = 0; i < nq * nq; ++i)
      cg_pinned(i) = shape_info.data.front().shape_gradients_collocation[i];
    for (int q = 0; q < nq; ++q)
      qw_pinned(q) = static_cast<number>(quadrature_1d.weight(q));

    Kokkos::deep_copy(comp_exec, this->shape_values, sv_pinned);
    Kokkos::deep_copy(comp_exec, this->co_shape_gradients, cg_pinned);
    Kokkos::deep_copy(comp_exec, this->quad_weights_1d, qw_pinned);

    const std::vector<unsigned int> &lex_numbering =
      shape_info.lexicographic_numbering;

    // 3. Build MPI-local constraint bitmask and owned-constrained index list
    std::vector<std::uint8_t> is_constrained_local(n_mpi_local_dofs, 0);
    std::vector<unsigned int> owned_constrained_host;
    owned_constrained_host.reserve(constraints.n_constraints());

    for (const auto &line : constraints.get_lines())
      {
        if (partitioner)
          {
            if (partitioner->in_local_range(line.index) ||
                partitioner->is_ghost_entry(line.index))
              {
                const unsigned int local_dof =
                  partitioner->global_to_local(line.index);
                if (local_dof < n_mpi_local_dofs)
                  {
                    is_constrained_local[local_dof] = 1;
                    if (local_dof < locally_owned_size)
                      owned_constrained_host.push_back(local_dof);
                  }
              }
          }
        else if (line.index < n_mpi_local_dofs)
          {
            const unsigned int local_dof =
              static_cast<unsigned int>(line.index);
            is_constrained_local[local_dof] = 1;
            owned_constrained_host.push_back(local_dof);
          }
      }

    this->owned_constrained_dofs =
      Kokkos::View<unsigned int *, internal::DeviceMemSpace>(
        Kokkos::view_alloc("opt2_owned_constrained_dofs",
                           Kokkos::WithoutInitializing),
        owned_constrained_host.size());
    Kokkos::View<unsigned int *, internal::PinnedHostMemSpace> constr_pinned;
    if (!owned_constrained_host.empty())
      {
        constr_pinned =
          Kokkos::View<unsigned int *, internal::PinnedHostMemSpace>(
            Kokkos::view_alloc("opt2_constr_pinned",
                               Kokkos::WithoutInitializing),
            owned_constrained_host.size());
        for (std::size_t i = 0; i < owned_constrained_host.size(); ++i)
          constr_pinned(i) = owned_constrained_host[i];
        Kokkos::deep_copy(comp_exec, this->owned_constrained_dofs, constr_pinned);
      }

    // 4. Partition locally owned active cells into:
    //    - 1 unified group when non-overlapped or single-rank
    //    - 2 groups when overlapped multi-GPU:
    //        color 0 = 100% of interior cells (unsplit! zero ghost DoFs)
    //        color 1 = thin boundary shell (cells with >= 1 unconstrained ghost DoF)
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    std::vector<CellIterator> owned_cells;
    owned_cells.reserve(parallel_tria ?
                          parallel_tria->n_locally_owned_active_cells() :
                          dof_handler.get_triangulation().n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        owned_cells.push_back(cell);

    std::vector<types::global_dof_index> cell_dofs(n_local_dofs);
    std::vector<std::vector<CellIterator>> colored_graph;
    std::vector<std::vector<unsigned int>> flat_lex_dofs_per_color;

    if (!owned_cells.empty())
      {
        if (has_ghost_exchange && overlap_communication_computation)
          {
            colored_graph.resize(2);
            flat_lex_dofs_per_color.resize(2);
            colored_graph[0].reserve(owned_cells.size());
            flat_lex_dofs_per_color[0].reserve(owned_cells.size() *
                                               n_local_dofs);

            std::array<unsigned int, n_local_dofs> mapped_dofs{};
            for (const auto &cell : owned_cells)
              {
                cell->get_dof_indices(cell_dofs);
                bool touches_unconstrained_ghost = false;
                for (unsigned int i = 0; i < n_local_dofs; ++i)
                  {
                    const types::global_dof_index gdof =
                      cell_dofs[lex_numbering[i]];
                    const unsigned int ldof = partitioner->global_to_local(gdof);
                    if (ldof < n_mpi_local_dofs && is_constrained_local[ldof])
                      {
                        mapped_dofs[i] = numbers::invalid_unsigned_int;
                      }
                    else
                      {
                        mapped_dofs[i] = ldof;
                        if (ldof >= locally_owned_size)
                          touches_unconstrained_ghost = true;
                      }
                  }

                const unsigned int target_color =
                  touches_unconstrained_ghost ? 1u : 0u;
                colored_graph[target_color].push_back(cell);
                flat_lex_dofs_per_color[target_color].insert(
                  flat_lex_dofs_per_color[target_color].end(),
                  mapped_dofs.begin(),
                  mapped_dofs.end());
              }
          }
        else
          {
            colored_graph.resize(1);
            flat_lex_dofs_per_color.resize(1);
            colored_graph[0] = std::move(owned_cells);
            flat_lex_dofs_per_color[0].resize(colored_graph[0].size() *
                                              n_local_dofs);

            for (std::size_t c = 0; c < colored_graph[0].size(); ++c)
              {
                colored_graph[0][c]->get_dof_indices(cell_dofs);
                unsigned int *dst_cell =
                  &flat_lex_dofs_per_color[0][c * n_local_dofs];
                for (unsigned int i = 0; i < n_local_dofs; ++i)
                  {
                    const types::global_dof_index gdof =
                      cell_dofs[lex_numbering[i]];
                    const unsigned int ldof =
                      partitioner ? partitioner->global_to_local(gdof) :
                                    static_cast<unsigned int>(gdof);
                    dst_cell[i] =
                      (ldof < n_mpi_local_dofs && is_constrained_local[ldof]) ?
                        numbers::invalid_unsigned_int :
                        ldof;
                  }
              }
          }
      }

    this->n_colors = colored_graph.size();
    this->n_cells_per_color.assign(n_colors, 0u);
    this->optimal_num_blocks_per_color.assign(n_colors, 0u);
    this->use_coloring_per_color.assign(n_colors, false);
    this->dirichlet_boundary_dofs_masks.clear();
    this->dirichlet_boundary_dofs_masks.resize(n_colors);
    this->G_tensors.clear();
    this->G_tensors.resize(n_colors);

    // 5. Double-buffered pinned-host async DMA + GPU G-tensor tile pipeline
    const MappingQ<dim> mapping(fe_degree);
    FEValues<dim>       fe_values(mapping,
                            dof_handler.get_fe(),
                            Quadrature<dim>(quadrature_1d),
                            update_inverse_jacobians | update_JxW_values);
    fe_values.always_allow_check_for_cell_similarity(true);

    constexpr int         symmetric_tensor_dim = (dim * (dim + 1)) / 2;
    constexpr std::size_t target_slot_bytes    = 4u * 1024u * 1024u;
    constexpr std::size_t bytes_per_cell_tile =
      static_cast<std::size_t>(n_q_points) * (dim * dim + 1u) * sizeof(number);
    constexpr unsigned int max_tile_cells = std::clamp<unsigned int>(
      static_cast<unsigned int>(target_slot_bytes / bytes_per_cell_tile),
      16u,
      4096u);

    using InvJacTileDev = Kokkos::
      View<number **[dim][dim], Kokkos::LayoutRight, internal::DeviceMemSpace>;
    using InvJacTilePinned = Kokkos::View<number **[dim][dim],
                                          Kokkos::LayoutRight,
                                          internal::PinnedHostMemSpace>;
    using JxWTileDev =
      Kokkos::View<number **, Kokkos::LayoutRight, internal::DeviceMemSpace>;
    using JxWTilePinned = Kokkos::
      View<number **, Kokkos::LayoutRight, internal::PinnedHostMemSpace>;

    const std::array<ExecutionSpace, 2> slot_exec = {{comp_exec, comm_exec}};

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();
        this->n_cells_per_color[color] = n_cells;
        this->optimal_num_blocks_per_color[color] =
          compute_optimal_num_blocks(n_cells);
        if (n_cells == 0u)
          continue;

        // 5a. Upload Dirichlet-masked lexicographic cell DoF table via pinned DMA
        this->dirichlet_boundary_dofs_masks[color] =
          Kokkos::View<unsigned int **, internal::DeviceMemSpace>(
            Kokkos::view_alloc("opt2_dirichlet_dofs_" + std::to_string(color),
                               Kokkos::WithoutInitializing),
            n_local_dofs,
            n_cells);

        using MaskLayout =
          typename Kokkos::View<unsigned int **,
                                internal::DeviceMemSpace>::array_layout;
        Kokkos::View<unsigned int **, MaskLayout, internal::PinnedHostMemSpace>
          mask_pinned(Kokkos::view_alloc("opt2_mask_pinned_" +
                                           std::to_string(color),
                                         Kokkos::WithoutInitializing),
                      n_local_dofs,
                      n_cells);

        const auto &flat_dofs = flat_lex_dofs_per_color[color];
        for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
          {
            const unsigned int *src_cell =
              &flat_dofs[static_cast<std::size_t>(cell_id) * n_local_dofs];
            for (unsigned int i = 0; i < n_local_dofs; ++i)
              mask_pinned(i, cell_id) = src_cell[i];
          }

        Kokkos::deep_copy(comp_exec,
                          this->dirichlet_boundary_dofs_masks[color],
                          mask_pinned);

        // 5b. Allocate G_tensors[color] and stream double-buffered pinned tiles
        this->G_tensors[color] =
          Kokkos::View<number *, internal::DeviceMemSpace>(
            Kokkos::view_alloc("opt2_G_tensor_color_" + std::to_string(color),
                               Kokkos::WithoutInitializing),
            static_cast<std::size_t>(symmetric_tensor_dim) * n_cells *
              n_q_points);

        const unsigned int tile_cap = std::min(n_cells, max_tile_cells);
        const unsigned int n_tiles  = (n_cells + tile_cap - 1u) / tile_cap;
        const unsigned int n_slots  = (n_tiles > 1u) ? 2u : 1u;

        std::array<InvJacTileDev, 2>    inv_jac_dev;
        std::array<InvJacTilePinned, 2> inv_jac_pinned;
        std::array<JxWTileDev, 2>       jxw_dev;
        std::array<JxWTilePinned, 2>    jxw_pinned;

        for (unsigned int s = 0; s < n_slots; ++s)
          {
            inv_jac_dev[s] =
              InvJacTileDev(Kokkos::view_alloc("opt2_inv_jac_dev_" +
                                                 std::to_string(s),
                                               Kokkos::WithoutInitializing),
                            tile_cap,
                            n_q_points);
            inv_jac_pinned[s] = InvJacTilePinned(
              Kokkos::view_alloc("opt2_inv_jac_pinned_" + std::to_string(s),
                                 Kokkos::WithoutInitializing),
              tile_cap,
              n_q_points);
            jxw_dev[s] =
              JxWTileDev(Kokkos::view_alloc("opt2_jxw_dev_" + std::to_string(s),
                                            Kokkos::WithoutInitializing),
                         tile_cap,
                         n_q_points);
            jxw_pinned[s] =
              JxWTilePinned(Kokkos::view_alloc("opt2_jxw_pinned_" +
                                                 std::to_string(s),
                                               Kokkos::WithoutInitializing),
                            tile_cap,
                            n_q_points);
          }

        for (unsigned int tile_idx = 0; tile_idx < n_tiles; ++tile_idx)
          {
            const unsigned int s           = tile_idx % n_slots;
            const unsigned int cell_offset = tile_idx * tile_cap;
            const unsigned int cur_tile =
              std::min(tile_cap, n_cells - cell_offset);

            // Wait ONLY on slot s if it was previously dispatched (tile_idx >= 2),
            // keeping the other slot's H2D DMA + GPU kernel overlapped with CPU FEValues!
            if (tile_idx >= n_slots)
              slot_exec[s].fence();

            auto &ij_host  = inv_jac_pinned[s];
            auto &jxw_host = jxw_pinned[s];

            for (unsigned int t = 0; t < cur_tile; ++t)
              {
                fe_values.reinit(colored_graph[color][cell_offset + t]);
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    jxw_host(t, q) = static_cast<number>(fe_values.JxW(q));
                    const auto &ij = fe_values.inverse_jacobian(q);
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = 0; d2 < dim; ++d2)
                        ij_host(t, q, d1, d2) =
                          static_cast<number>(ij[d1][d2]);
                  }
              }

            const auto tile_range = std::make_pair(0u, cur_tile);
            auto       ij_dev_sub =
              Kokkos::subview(inv_jac_dev[s],
                              tile_range,
                              Kokkos::ALL,
                              Kokkos::ALL,
                              Kokkos::ALL);
            auto ij_host_sub =
              Kokkos::subview(ij_host,
                              tile_range,
                              Kokkos::ALL,
                              Kokkos::ALL,
                              Kokkos::ALL);
            auto jxw_dev_sub =
              Kokkos::subview(jxw_dev[s], tile_range, Kokkos::ALL);
            auto jxw_host_sub =
              Kokkos::subview(jxw_host, tile_range, Kokkos::ALL);

            Kokkos::deep_copy(slot_exec[s], ij_dev_sub, ij_host_sub);
            Kokkos::deep_copy(slot_exec[s], jxw_dev_sub, jxw_host_sub);

            const std::size_t total_q_work =
              static_cast<std::size_t>(cur_tile) * n_q_points;
            Kokkos::parallel_for(
              "PortableOpt2::FillGTensorTile",
              Kokkos::RangePolicy<ExecutionSpace>(slot_exec[s],
                                                  std::size_t(0),
                                                  total_q_work),
              internal::FillGTensorTileCoalescedFunctor<
                dim,
                nq,
                number,
                InvJacTileDev,
                JxWTileDev,
                Kokkos::View<number *, internal::DeviceMemSpace>>{
                cell_offset,
                inv_jac_dev[s],
                jxw_dev[s],
                this->G_tensors[color]});
          }

        comp_exec.fence();
        if (n_slots > 1u)
          comm_exec.fence();
      }

    comp_exec.fence();
    comm_exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::apply_constrained_values(
    const VectorType     &src,
    VectorType           &dst,
    const ExecutionSpace &exec,
    const bool            add_mode) const
  {
    const unsigned int n_constr = owned_constrained_dofs.extent(0);
    if (n_constr == 0u)
      return;

    if (add_mode)
      {
        Kokkos::parallel_for(
          "PortableOpt2::add_constrained_values",
          Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_constr),
          internal::ApplyConstrainedValuesFunctor<number, true>{
            owned_constrained_dofs, src.get_values(), dst.get_values()});
      }
    else
      {
        Kokkos::parallel_for(
          "PortableOpt2::copy_constrained_values",
          Kokkos::RangePolicy<ExecutionSpace>(exec, 0u, n_constr),
          internal::ApplyConstrainedValuesFunctor<number, false>{
            owned_constrained_dofs, src.get_values(), dst.get_values()});
      }
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::launch_color_on_stream(
    const ExecutionSpace                 &exec,
    const unsigned int                    color,
    const Portable::DeviceVector<number> &src_device,
    Portable::DeviceVector<number>       &dst_device) const
  {
    if (color >= n_colors)
      return;
    const unsigned int n_cells = n_cells_per_color[color];
    if (n_cells == 0u)
      return;

    constexpr bool is_serial =
      std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>;
    const unsigned int threads_per_block =
      is_serial ? 1u : numbers::invalid_unsigned_int;

    BK3Opt::Parallel::KokkosKernelAsync<dim, fe_degree + 1, nq, number>(
      exec,
      shape_values,
      co_shape_gradients,
      G_tensors[color],
      src_device,
      dst_device,
      dirichlet_boundary_dofs_masks[color],
      n_cells,
      optimal_num_blocks_per_color[color],
      threads_per_block,
      use_coloring_per_color[color]);
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::vmult_async(
    const ExecutionSpace &caller_exec,
    VectorType           &dst,
    const VectorType     &src) const
  {
    AssertDimension(dst.size(), src.size());

    const unsigned int n_owned = src.locally_owned_size();
    const unsigned int n_ghost = has_ghost_exchange ? ghost_size : 0u;
    const unsigned int total_local = n_owned + n_ghost;

    number       *src_mutable_ptr = const_cast<number *>(src.get_values());
    number       *dst_ptr         = dst.get_values();
    const number *src_owned_ptr   = src_mutable_ptr;
    number       *src_ghost_ptr   = src_mutable_ptr + n_owned;
    number       *dst_owned_ptr   = dst_ptr;
    number       *dst_ghost_ptr   = dst_ptr + n_owned;

    Portable::DeviceVector<number> src_device(src_mutable_ptr, total_local);
    Portable::DeviceVector<number> dst_device(dst_ptr, total_local);

    Kokkos::View<number *,
                 internal::DeviceMemSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      dst_owned_view(dst_owned_ptr, n_owned);

    // Case 1: Single-rank / no MPI halo exchange — pure async on caller_exec
    if (!has_ghost_exchange)
      {
        Kokkos::deep_copy(caller_exec, dst_owned_view, number(0));
        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color_on_stream(caller_exec, color, src_device, dst_device);
        apply_constrained_values(src, dst, caller_exec, /*add_mode=*/false);
        return;
      }

    // Case 2: Multi-GPU overlapped execution across caller_exec (interior) &
    //         high-priority comm_exec (halo MPI + boundary shell)
    if (overlap_communication_computation)
      {
        const ExecutionSpace &comm_exec = stream_ctx.get_comm_exec();
        const bool            has_boundary_cells =
          (n_colors > 1u && n_cells_per_color[1] > 0u);

        // Ensure any prior async writes to src on caller_exec are visible to comm_exec
        caller_ready_event.record(caller_exec);
        caller_ready_event.wait(comm_exec);

        // Kick off dst_owned zero + 100% interior compute on caller_exec BEFORE
        // comm_exec.fence() inside start_forward so GPU SMs stay busy continuously!
        Kokkos::deep_copy(caller_exec, dst_owned_view, number(0));
        dst_zeroed_event.record(caller_exec);
        if (n_colors > 0u)
          launch_color_on_stream(caller_exec, 0u, src_device, dst_device);

        // Forward halo pack (+ zero dst_ghost if boundary cells exist) + MPI Isend/Irecv
        halo_exchanger.start_forward(
          comm_exec,
          src_owned_ptr,
          src_ghost_ptr,
          has_boundary_cells ? dst_ghost_ptr :
                               (ghost_size > 0u ? dst_ghost_ptr : nullptr),
          0u);
        halo_exchanger.finish_forward(comm_exec, src_ghost_ptr);

        // Guarantee dst_owned zeroing on caller_exec finished before boundary
        // cells or reverse halo unpack atomic_add into dst_owned!
        dst_zeroed_event.wait(comm_exec);

        if (has_boundary_cells)
          launch_color_on_stream(comm_exec, 1u, src_device, dst_device);

        // Reverse halo exchange + 1-kernel atomic unpack on comm_exec
        halo_exchanger.start_reverse(comm_exec, dst_ghost_ptr, 0u);
        halo_exchanger.finish_reverse_async(comm_exec,
                                            dst_owned_ptr,
                                            src_ghost_ptr,
                                            dst_ghost_ptr);

        // Join comm_exec back onto caller_exec via GPU event (zero host stall)
        comm_done_event.record(comm_exec);
        comm_done_event.wait(caller_exec);

        apply_constrained_values(src, dst, caller_exec, /*add_mode=*/false);
        return;
      }

    // Case 3: Multi-GPU non-overlapped mode (--no-overlap) on caller_exec
    Kokkos::deep_copy(caller_exec, dst_owned_view, number(0));
    halo_exchanger.start_forward(caller_exec,
                                 src_owned_ptr,
                                 src_ghost_ptr,
                                 (ghost_size > 0u ? dst_ghost_ptr : nullptr),
                                 0u);
    halo_exchanger.finish_forward(caller_exec, src_ghost_ptr);

    for (unsigned int color = 0; color < n_colors; ++color)
      launch_color_on_stream(caller_exec, color, src_device, dst_device);

    halo_exchanger.start_reverse(caller_exec, dst_ghost_ptr, 0u);
    halo_exchanger.finish_reverse_async(caller_exec,
                                        dst_owned_ptr,
                                        src_ghost_ptr,
                                        dst_ghost_ptr);
    apply_constrained_values(src, dst, caller_exec, /*add_mode=*/false);
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    const ExecutionSpace &comp_exec = stream_ctx.get_compute_exec();
    vmult_async(comp_exec, dst, src);
    comp_exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::vmult_add(
    VectorType       &dst,
    const VectorType &src) const
  {
    AssertDimension(dst.size(), src.size());

    const ExecutionSpace &comp_exec   = stream_ctx.get_compute_exec();
    const unsigned int    n_owned     = src.locally_owned_size();
    const unsigned int    n_ghost     = has_ghost_exchange ? ghost_size : 0u;
    const unsigned int    total_local = n_owned + n_ghost;

    number       *src_mutable_ptr = const_cast<number *>(src.get_values());
    number       *dst_ptr         = dst.get_values();
    const number *src_owned_ptr   = src_mutable_ptr;
    number       *src_ghost_ptr   = src_mutable_ptr + n_owned;
    number       *dst_owned_ptr   = dst_ptr;
    number       *dst_ghost_ptr   = dst_ptr + n_owned;

    Portable::DeviceVector<number> src_device(src_mutable_ptr, total_local);
    Portable::DeviceVector<number> dst_device(dst_ptr, total_local);

    if (!has_ghost_exchange)
      {
        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color_on_stream(comp_exec, color, src_device, dst_device);
        apply_constrained_values(src, dst, comp_exec, /*add_mode=*/true);
        comp_exec.fence();
        return;
      }

    halo_exchanger.start_forward(comp_exec,
                                 src_owned_ptr,
                                 src_ghost_ptr,
                                 (ghost_size > 0u ? dst_ghost_ptr : nullptr),
                                 0u);
    halo_exchanger.finish_forward(comp_exec, src_ghost_ptr);

    for (unsigned int color = 0; color < n_colors; ++color)
      launch_color_on_stream(comp_exec, color, src_device, dst_device);

    halo_exchanger.start_reverse(comp_exec, dst_ghost_ptr, 0u);
    halo_exchanger.finish_reverse_async(comp_exec,
                                        dst_owned_ptr,
                                        src_ghost_ptr,
                                        dst_ghost_ptr);
    apply_constrained_values(src, dst, comp_exec, /*add_mode=*/true);
    comp_exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::vmult_dummy(
    VectorType       &dst,
    const VectorType &src,
    const bool        ghost_exchange_on,
    const bool        computation_on) const
  {
    if (ghost_exchange_on && computation_on)
      {
        vmult(dst, src);
        return;
      }

    const ExecutionSpace &comp_exec   = stream_ctx.get_compute_exec();
    const ExecutionSpace &comm_exec   = stream_ctx.get_comm_exec();
    const unsigned int    n_owned     = src.locally_owned_size();
    const unsigned int    n_ghost     = has_ghost_exchange ? ghost_size : 0u;
    const unsigned int    total_local = n_owned + n_ghost;

    number       *src_mutable_ptr = const_cast<number *>(src.get_values());
    number       *dst_ptr         = dst.get_values();
    const number *src_owned_ptr   = src_mutable_ptr;
    number       *src_ghost_ptr   = src_mutable_ptr + n_owned;
    number       *dst_owned_ptr   = dst_ptr;
    number       *dst_ghost_ptr   = dst_ptr + n_owned;

    if (ghost_exchange_on && !computation_on)
      {
        if (has_ghost_exchange)
          {
            halo_exchanger.start_forward(
              comm_exec,
              src_owned_ptr,
              src_ghost_ptr,
              (ghost_size > 0u ? dst_ghost_ptr : nullptr),
              0u);
            halo_exchanger.finish_forward(comm_exec, src_ghost_ptr);
            halo_exchanger.start_reverse(comm_exec, dst_ghost_ptr, 0u);
            halo_exchanger.finish_reverse_async(comm_exec,
                                                dst_owned_ptr,
                                                src_ghost_ptr,
                                                dst_ghost_ptr);
            comm_exec.fence();
          }
        return;
      }

    if (computation_on)
      {
        Portable::DeviceVector<number> src_device(src_mutable_ptr, total_local);
        Portable::DeviceVector<number> dst_device(dst_ptr, total_local);

        Kokkos::View<number *,
                     internal::DeviceMemSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          dst_full_view(dst_ptr, total_local);
        Kokkos::deep_copy(comp_exec, dst_full_view, number(0));

        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color_on_stream(comp_exec, color, src_device, dst_device);

        if constexpr (running_in_debug_mode())
          {
            if (n_ghost > 0u)
              {
                Kokkos::View<number *,
                             internal::DeviceMemSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                  dst_ghost_view(dst_ghost_ptr, n_ghost);
                Kokkos::deep_copy(comp_exec, dst_ghost_view, number(0));
              }
          }

        apply_constrained_values(src, dst, comp_exec, /*add_mode=*/false);
        comp_exec.fence();
      }
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::assemble_constant_rhs(
    VectorType  &dst,
    const number rhs_value) const
  {
    AssertDimension(dst.size(), m());

    const ExecutionSpace &comp_exec = stream_ctx.get_compute_exec();
    dst                             = number(0);

    const unsigned int n_owned     = dst.locally_owned_size();
    const unsigned int n_ghost     = has_ghost_exchange ? ghost_size : 0u;
    const unsigned int total_local = n_owned + n_ghost;

    number *dst_ptr       = dst.get_values();
    number *dst_owned_ptr = dst_ptr;
    number *dst_ghost_ptr = dst_ptr + n_owned;

    if (n_ghost > 0u)
      {
        Kokkos::View<number *,
                     internal::DeviceMemSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          dst_ghost_view(dst_ghost_ptr, n_ghost);
        Kokkos::deep_copy(comp_exec, dst_ghost_view, number(0));
      }

    BK3Opt::Parallel::DeviceView<number> dst_device(dst_ptr, total_local);

    constexpr unsigned int nm       = fe_degree + 1;
    constexpr unsigned int nq_total = Utilities::pow(nq, dim);
    using SharedView                = BK3Opt::Parallel::SharedView<number>;
    using TeamPolicy                = Kokkos::TeamPolicy<ExecutionSpace>;

    constexpr std::size_t shmem_budget_bytes = 16384;
    const std::size_t     fixed_shmem =
      SharedView::shmem_size(nm * nq) + SharedView::shmem_size(nq);
    const std::size_t per_elem_bytes = 2 * SharedView::shmem_size(nq_total);

    unsigned int nelmtPerBatch = 1u;
    if (shmem_budget_bytes > fixed_shmem)
      nelmtPerBatch = std::max<unsigned int>(
        1u,
        static_cast<unsigned int>((shmem_budget_bytes - fixed_shmem) /
                                  per_elem_bytes));
    nelmtPerBatch = std::min(nelmtPerBatch, std::max(1u, 256u / (nq * nq)));

    const std::size_t shmem_size =
      fixed_shmem + 2 * SharedView::shmem_size(nelmtPerBatch * nq_total);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = n_cells_per_color[color];
        if (n_cells == 0u)
          continue;

        const unsigned int total_batches =
          (n_cells + nelmtPerBatch - 1u) / nelmtPerBatch;

        internal::ConstantRhsKernelFunctorOpt2<dim, nm, nq, number> functor{
          shape_values,
          quad_weights_1d,
          G_tensors[color],
          dst_device,
          dirichlet_boundary_dofs_masks[color],
          rhs_value,
          n_cells,
          nelmtPerBatch,
          total_batches,
          shmem_size,
          use_coloring_per_color[color]};

        constexpr bool is_serial =
          std::is_same_v<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>;

        const unsigned int numBlocks = is_serial ? 1u : total_batches;
        unsigned int       threadsPerBlock =
          is_serial ? 1u : (nq * nq * nelmtPerBatch);

        TeamPolicy         probe_policy(comp_exec, 1, Kokkos::AUTO);
        const unsigned int max_team_size = static_cast<unsigned int>(
          probe_policy.team_size_max(functor, Kokkos::ParallelForTag()));
        threadsPerBlock =
          std::max(1u, std::min(threadsPerBlock, max_team_size));

        TeamPolicy policy(comp_exec, numBlocks, threadsPerBlock);
        Kokkos::parallel_for(
          "LaplaceOperatorOpt2::assemble_constant_rhs", policy, functor);
      }

    if (has_ghost_exchange)
      {
        halo_exchanger.start_reverse(comp_exec, dst_ghost_ptr, 1u);
        halo_exchanger.finish_reverse_async(comp_exec,
                                            dst_owned_ptr,
                                            nullptr,
                                            dst_ghost_ptr);
      }
    comp_exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::Tvmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == partitioner,
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == partitioner,
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::initialize_dof_vector(
    VectorType &vec) const
  {
    if (partitioner)
      vec.reinit(partitioner);
    else
      vec.reinit(total_dofs);
  }

  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::m() const
  {
    return partitioner ? partitioner->size() : total_dofs;
  }

  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::n() const
  {
    return partitioner ? partitioner->size() : total_dofs;
  }

  template <int dim, int fe_degree, int nq, typename number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::get_vector_partitioner()
    const
  {
    return partitioner;
  }

  template <int dim, int fe_degree, int nq, typename number>
  BK3Opt::Parallel::ConstDeviceView<number>
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::get_G_tensor(
    const unsigned int color) const
  {
    AssertIndexRange(color, G_tensors.size());
    return G_tensors[color];
  }

  template <int dim, int fe_degree, int nq, typename number>
  std::size_t
  LaplaceOperatorOpt2<dim, fe_degree, nq, number>::memory_consumption() const
  {
    std::size_t bytes =
      (shape_values.size() + co_shape_gradients.size() +
       quad_weights_1d.size()) *
        sizeof(number) +
      owned_constrained_dofs.size() * sizeof(unsigned int) +
      static_cast<std::size_t>(halo_exchanger.n_import_indices()) *
        (sizeof(unsigned int) + 2u * sizeof(number));

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        bytes +=
          dirichlet_boundary_dofs_masks[color].size() * sizeof(unsigned int) +
          G_tensors[color].size() * sizeof(number);
      }

    return bytes;
  }

} // namespace PortableOpt2

#endif // portable_laplace_operator_opt2_h
