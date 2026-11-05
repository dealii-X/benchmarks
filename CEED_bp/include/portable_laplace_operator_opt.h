#ifndef portable_laplace_operator_opt_h
#define portable_laplace_operator_opt_h

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
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "bk3_kokkos_kernel_opt.h"

DEAL_II_NAMESPACE_OPEN

namespace PortableOpt
{
  namespace internal
  {
    template <int dim, int nm, int nq, typename number>
    struct ConstantRhsKernelFunctor
    {
      using ExecutionSpace = BK3Opt::Parallel::ExecutionSpace;
      using TeamPolicy     = Kokkos::TeamPolicy<ExecutionSpace>;
      using MemberType     = typename TeamPolicy::member_type;
      using SharedView     = BK3Opt::Parallel::SharedView<number>;

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

        for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
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

            // Step 1: Recover exact JxW = det(G) / w_3d^2 at quadrature points
            // from G-tensor and contract along direction 0 (r -> k)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * nq * nq;
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

            // Step 2: Contract along direction 1 (q -> j)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * nm * nq;
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

            // Step 3: Contract along direction 2 (p -> i) & scatter-accumulate
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * nm * nm;
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
              typename ExecutionSpace,
              typename InvJacView,
              typename JxWView,
              typename GView>
    void
    fill_G_tensor_tile_gpu(const ExecutionSpace &exec,
                           const unsigned int    color,
                           const unsigned int    /*total_cells_in_color*/,
                           const unsigned int    cell_offset,
                           const unsigned int    tile_cells,
                           const InvJacView     &inv_jacobian_tile,
                           const JxWView        &JxW_tile,
                           GView                &G)
    {
      constexpr unsigned int n_q_points = Utilities::pow(nq, dim);
      constexpr int          symmetric_tensor_dim = (dim * (dim + 1)) / 2;

      Kokkos::parallel_for(
        "Fill_G_tensor_tile_color_" + std::to_string(color),
        Kokkos::RangePolicy<ExecutionSpace>(exec, 0, tile_cells),
        KOKKOS_LAMBDA(const unsigned int t) {
          const unsigned int cell_id = cell_offset + t;
          const std::size_t  g_cell_base =
            static_cast<std::size_t>(cell_id) * symmetric_tensor_dim *
            n_q_points;
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const number jxw_val = JxW_tile(q_point, t);

              int c = 0;
              for (int d1 = 0; d1 < dim; ++d1)
                for (int d2 = d1; d2 < dim; ++d2)
                  {
                    number sum = 0;
                    for (int k = 0; k < dim; ++k)
                      sum += inv_jacobian_tile(q_point, t, d1, k) *
                             inv_jacobian_tile(q_point, t, d2, k);
                    G[g_cell_base + static_cast<std::size_t>(c) * n_q_points +
                      q_point] = jxw_val * sum;
                    ++c;
                  }
            }
        });
    }

    template <typename ExecutionSpace, typename number>
    void
    copy_constrained_values_gpu(
      const ExecutionSpace &exec,
      const Kokkos::View<const unsigned int *,
                         MemorySpace::Default::kokkos_space> &constr_dofs,
      const number                                           *src_ptr,
      number                                                 *dst_ptr)
    {
      const unsigned int n_constr = constr_dofs.extent(0);
      if (n_constr == 0)
        return;

      Kokkos::parallel_for(
        "LaplaceOperatorOpt::copy_constrained_values",
        Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n_constr),
        KOKKOS_LAMBDA(const unsigned int i) {
          const unsigned int dof = constr_dofs(i);
          dst_ptr[dof]           = src_ptr[dof];
        });
    }
  } // namespace internal

  template <int dim, int fe_degree, int nq, typename number>
  class LaplaceOperatorOpt
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>;

    LaplaceOperatorOpt(const DoFHandler<dim>           &dof_handler,
                       const AffineConstraints<number> &constraints,
                       bool overlap_communication_computation);

    void
    vmult(VectorType &dst, const VectorType &src) const;

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

    BK3Opt::Parallel::ConstDeviceView<number>
    get_G_tensor(const unsigned int color = 0) const;

    std::size_t
    memory_consumption() const;

  private:
    using ExecutionSpace =
      dealii::MemorySpace::Default::kokkos_space::execution_space;

    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);

    static constexpr unsigned int n_q_points = Utilities::pow(nq, dim);

    void
    setup_operator(const DoFHandler<dim>           &dof_handler,
                   const AffineConstraints<number> &constraints);

    void
    copy_constrained_values(const VectorType     &src,
                            VectorType           &dst,
                            const ExecutionSpace &exec) const;

    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
    types::global_dof_index                            total_dofs         = 0;
    unsigned int                                       locally_owned_size = 0;
    unsigned int                                       ghost_size         = 0;
    bool has_ghost_exchange                = false;
    bool overlap_communication_computation = false;

    unsigned int              n_colors = 0;
    std::vector<unsigned int> n_cells_per_color;
    std::vector<bool>         use_coloring_per_color;

    Kokkos::View<number *, MemorySpace::Default::kokkos_space> shape_values;
    Kokkos::View<number *, MemorySpace::Default::kokkos_space>
      co_shape_gradients;
    Kokkos::View<number *, MemorySpace::Default::kokkos_space> quad_weights_1d;

    Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>
      owned_constrained_dofs;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dirichlet_boundary_dofs_masks;

    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>>
      G_tensors;
  };

  template <int dim, int fe_degree, int nq, typename number>
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::LaplaceOperatorOpt(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
    : total_dofs(dof_handler.n_dofs())
    , overlap_communication_computation(overlap_communication_computation)
  {
    // Validate up front in O(N_constrained) that all constraints are
    // homogeneous Dirichlet.
    for (const auto &line : constraints.get_lines())
      {
        AssertThrow(
          line.entries.empty() && line.inhomogeneity == number(0),
          ExcMessage(
            "LaplaceOperatorOpt only supports homogeneous Dirichlet constraints; "
            "hanging-node or inhomogeneous constraints were detected."));
      }

    setup_operator(dof_handler, constraints);
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::setup_operator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints)
  {
    AssertDimension(dim, 3);
    AssertDimension(dof_handler.get_fe().n_dofs_per_cell(), n_local_dofs);
    ExecutionSpace exec;

    // 1. Setup MPI Partitioner directly from DoFHandler
    const auto *parallel_tria =
      dynamic_cast<const parallel::TriangulationBase<dim> *>(
        &dof_handler.get_triangulation());
    if (parallel_tria != nullptr)
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
      partitioner && (partitioner->n_ghost_indices() > 0 ||
                      partitioner->n_import_indices() > 0);

    const std::size_t n_mpi_local_dofs =
      static_cast<std::size_t>(locally_owned_size) + ghost_size;
    AssertThrow(
      n_mpi_local_dofs <
        static_cast<std::size_t>(numbers::invalid_unsigned_int),
      ExcMessage("Local DoF index exceeds 32-bit unsigned int range."));

    // 2. Build 1D shape tables, collocation derivatives, lexicographic map &
    //    quadrature weights directly via ShapeInfo (zero full-mesh MatrixFree bloat)
    const QGauss<1> quadrature_1d(nq);
    const dealii::internal::MatrixFreeFunctions::ShapeInfo<number> shape_info(
      quadrature_1d, dof_handler.get_fe());

    constexpr unsigned int n_dofs_1d    = fe_degree + 1;
    const unsigned int     n_shape_vals = n_dofs_1d * nq;

    this->shape_values =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("shape_values", Kokkos::WithoutInitializing),
        n_shape_vals);
    {
      auto sv_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                  this->shape_values);
      for (unsigned int i = 0; i < n_shape_vals; ++i)
        sv_mirror(i) = shape_info.data.front().shape_values[i];
      Kokkos::deep_copy(exec, this->shape_values, sv_mirror);
    }

    this->co_shape_gradients =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("co_shape_gradients", Kokkos::WithoutInitializing),
        nq * nq);
    {
      auto cg_mirror = Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                                  this->co_shape_gradients);
      for (int i = 0; i < nq * nq; ++i)
        cg_mirror(i) = shape_info.data.front().shape_gradients_collocation[i];
      Kokkos::deep_copy(exec, this->co_shape_gradients, cg_mirror);
    }

    this->quad_weights_1d =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("quad_weights_1d", Kokkos::WithoutInitializing), nq);
    {
      auto weights_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, quad_weights_1d);
      for (int q = 0; q < nq; ++q)
        weights_host(q) = static_cast<number>(quadrature_1d.weight(q));
      Kokkos::deep_copy(exec, quad_weights_1d, weights_host);
    }

    const std::vector<unsigned int> &lex_numbering =
      shape_info.lexicographic_numbering;

    // 3. Build MPI-local constraint bitmask and owned-constrained index list
    //    in O(N_constrained)
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
      Kokkos::View<unsigned int *, MemorySpace::Default::kokkos_space>(
        Kokkos::view_alloc("owned_constrained_dofs",
                           Kokkos::WithoutInitializing),
        owned_constrained_host.size());
    if (!owned_constrained_host.empty())
      {
        Kokkos::View<const unsigned int *,
                     Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          owned_host_view(owned_constrained_host.data(),
                          owned_constrained_host.size());
        auto owned_mirror = Kokkos::create_mirror_view(
          Kokkos::WithoutInitializing, this->owned_constrained_dofs);
        Kokkos::deep_copy(owned_mirror, owned_host_view);
        Kokkos::deep_copy(exec, this->owned_constrained_dofs, owned_mirror);
      }

    // 4. Partition locally owned active cells into 1 color (non-overlapped) or
    //    3 colors (interior_0, boundary_1, interior_2 for non-blocking MPI overlap)
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    std::vector<CellIterator> owned_cells;
    owned_cells.reserve(parallel_tria ?
                          parallel_tria->n_locally_owned_active_cells() :
                          dof_handler.get_triangulation().n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        owned_cells.push_back(cell);

    std::vector<std::vector<CellIterator>> colored_graph;
    if (!owned_cells.empty())
      {
        if (overlap_communication_computation)
          {
            colored_graph.resize(3);
            std::vector<bool> ghost_vertices(
              dof_handler.get_triangulation().n_vertices(), false);
            for (const auto &cell :
                 dof_handler.get_triangulation().active_cell_iterators())
              if (cell->is_ghost())
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     ++v)
                  ghost_vertices[cell->vertex_index(v)] = true;

            std::vector<CellIterator> inner_cells;
            inner_cells.reserve(owned_cells.size());
            for (const auto &cell : owned_cells)
              {
                bool touches_ghost = false;
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     ++v)
                  if (ghost_vertices[cell->vertex_index(v)])
                    {
                      touches_ghost = true;
                      break;
                    }
                if (touches_ghost)
                  colored_graph[1].push_back(cell);
                else
                  inner_cells.push_back(cell);
              }
            const std::size_t half = inner_cells.size() / 2;
            colored_graph[0].assign(inner_cells.begin(),
                                    inner_cells.begin() + half);
            colored_graph[2].assign(inner_cells.begin() + half,
                                    inner_cells.end());
          }
        else
          {
            colored_graph.resize(1);
            colored_graph[0] = std::move(owned_cells);
          }
      }

    this->n_colors = colored_graph.size();
    this->n_cells_per_color.assign(n_colors, 0u);
    this->use_coloring_per_color.assign(n_colors, false);
    this->dirichlet_boundary_dofs_masks.clear();
    this->dirichlet_boundary_dofs_masks.resize(n_colors);
    this->G_tensors.clear();
    this->G_tensors.resize(n_colors);

    // 5. Evaluate MappingQ(fe_degree) inverse Jacobians & JxW ONLY (skipping
    //    all wasted 3D shape_values/shape_gradients/quadrature_points work in
    //    Portable::MatrixFree::reinit) and stream in small tiles so peak GPU
    //    VRAM == steady-state GPU VRAM (~68% total & peak VRAM reduction!).
    const MappingQ<dim> mapping(fe_degree);
    FEValues<dim>       fe_values(mapping,
                            dof_handler.get_fe(),
                            Quadrature<dim>(quadrature_1d),
                            update_inverse_jacobians | update_JxW_values);
    fe_values.always_allow_check_for_cell_similarity(true);

    constexpr int                        symmetric_tensor_dim = (dim * (dim + 1)) / 2;
    constexpr unsigned int               max_tile_cells       = 4096u;
    std::vector<types::global_dof_index> cell_dofs(n_local_dofs);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();
        this->n_cells_per_color[color] = n_cells;
        if (n_cells == 0)
          continue;

        // 5a. Build Dirichlet-masked lexicographic cell DoF table directly
        this->dirichlet_boundary_dofs_masks[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("dirichlet_boundary_dofs_" +
                                 std::to_string(color),
                               Kokkos::WithoutInitializing),
            n_local_dofs,
            n_cells);

        {
          auto mask_host = Kokkos::create_mirror_view(
            Kokkos::WithoutInitializing,
            this->dirichlet_boundary_dofs_masks[color]);

          for (unsigned int cell_id = 0; cell_id < n_cells; ++cell_id)
            {
              colored_graph[color][cell_id]->get_dof_indices(cell_dofs);
              for (unsigned int i = 0; i < n_local_dofs; ++i)
                {
                  const types::global_dof_index gdof =
                    cell_dofs[lex_numbering[i]];
                  const unsigned int ldof =
                    partitioner ? partitioner->global_to_local(gdof) :
                                  static_cast<unsigned int>(gdof);
                  mask_host(i, cell_id) =
                    (ldof < n_mpi_local_dofs && is_constrained_local[ldof]) ?
                      numbers::invalid_unsigned_int :
                      ldof;
                }
            }

          Kokkos::deep_copy(
            exec, this->dirichlet_boundary_dofs_masks[color], mask_host);
        }

        // 5b. Allocate steady-state G_tensors[color] and stream geometry tiles
        this->G_tensors[color] =
          Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("G_tensor_color_" + std::to_string(color),
                               Kokkos::WithoutInitializing),
            static_cast<std::size_t>(symmetric_tensor_dim) * n_cells *
              n_q_points);

        const unsigned int tile_cap = std::min(n_cells, max_tile_cells);
        Kokkos::View<number **[dim][dim], MemorySpace::Default::kokkos_space>
          inv_jac_tile_dev(Kokkos::view_alloc("inv_jac_tile",
                                              Kokkos::WithoutInitializing),
                           n_q_points,
                           tile_cap);
        Kokkos::View<number **, MemorySpace::Default::kokkos_space>
          jxw_tile_dev(Kokkos::view_alloc("jxw_tile",
                                          Kokkos::WithoutInitializing),
                       n_q_points,
                       tile_cap);

        auto inv_jac_tile_host = Kokkos::create_mirror_view(
          Kokkos::WithoutInitializing, inv_jac_tile_dev);
        auto jxw_tile_host =
          Kokkos::create_mirror_view(Kokkos::WithoutInitializing, jxw_tile_dev);

        for (unsigned int cell_offset = 0; cell_offset < n_cells;
             cell_offset += tile_cap)
          {
            const unsigned int cur_tile =
              std::min(tile_cap, n_cells - cell_offset);

            for (unsigned int t = 0; t < cur_tile; ++t)
              {
                fe_values.reinit(colored_graph[color][cell_offset + t]);
                for (unsigned int q = 0; q < n_q_points; ++q)
                  {
                    jxw_tile_host(q, t) =
                      static_cast<number>(fe_values.JxW(q));
                    const auto &ij = fe_values.inverse_jacobian(q);
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = 0; d2 < dim; ++d2)
                        inv_jac_tile_host(q, t, d1, d2) =
                          static_cast<number>(ij[d1][d2]);
                  }
              }

            Kokkos::deep_copy(exec, inv_jac_tile_dev, inv_jac_tile_host);
            Kokkos::deep_copy(exec, jxw_tile_dev, jxw_tile_host);

            internal::fill_G_tensor_tile_gpu<dim, nq, number>(
              exec,
              color,
              n_cells,
              cell_offset,
              cur_tile,
              inv_jac_tile_dev,
              jxw_tile_dev,
              this->G_tensors[color]);
            exec.fence();
          }
      }

    exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::copy_constrained_values(
    const VectorType     &src,
    VectorType           &dst,
    const ExecutionSpace &exec) const
  {
    internal::copy_constrained_values_gpu(exec,
                                          owned_constrained_dofs,
                                          src.get_values(),
                                          dst.get_values());
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    AssertDimension(dst.size(), src.size());

    ExecutionSpace exec;

    if (dst.has_ghost_elements())
      dst.zero_out_ghost_values();

    const auto src_local_size =
      src.locally_owned_size() +
      (has_ghost_exchange ? src.get_partitioner()->n_ghost_indices() : 0u);
    const auto dst_local_size =
      dst.locally_owned_size() +
      (has_ghost_exchange ? dst.get_partitioner()->n_ghost_indices() : 0u);

    Portable::DeviceVector<number> src_device(src.get_values(), src_local_size),
      dst_device(dst.get_values(), dst_local_size);

    Kokkos::deep_copy(exec, dst_device, number(0));

    auto launch_color = [&](const unsigned int color) {
      const unsigned int n_cells = n_cells_per_color[color];
      if (n_cells == 0)
        return;

      constexpr bool is_serial =
        std::is_same<Kokkos::DefaultExecutionSpace,
                     Kokkos::DefaultHostExecutionSpace>::value;

      const unsigned int numBlocks =
        is_serial ? 1u : numbers::invalid_unsigned_int;
      const unsigned int threadsPerBlock =
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
        numBlocks,
        threadsPerBlock,
        use_coloring_per_color[color]);
    };

    if (!has_ghost_exchange)
      {
        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color(color);

        copy_constrained_values(src, dst, exec);
        exec.fence();
        return;
      }

    if (overlap_communication_computation && n_colors == 3)
      {
        // 1. Start non-blocking ghost exchange overlapped with interior batch 0
        src.update_ghost_values_start(0);
        launch_color(0);
        src.update_ghost_values_finish();

        // 2. Evaluate boundary cells (color 1) now that ghosts have arrived
        if (n_cells_per_color[1] > 0)
          {
            launch_color(1);
            exec.fence();
          }

        // 3. Start non-blocking ghost reduction overlapped with interior batch 2
        dst.compress_start(0, VectorOperation::add);
        launch_color(2);
        dst.compress_finish(VectorOperation::add);

        src.zero_out_ghost_values();
        copy_constrained_values(src, dst, exec);
        exec.fence();
      }
    else
      {
        src.update_ghost_values();

        for (unsigned int color = 0; color < n_colors; ++color)
          launch_color(color);

        exec.fence();
        dst.compress(VectorOperation::add);
        src.zero_out_ghost_values();
        copy_constrained_values(src, dst, exec);
        exec.fence();
      }
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::vmult_dummy(
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

    ExecutionSpace exec;

    if (ghost_exchange_on && !computation_on)
      {
        if (has_ghost_exchange)
          {
            if (dst.has_ghost_elements())
              dst.zero_out_ghost_values();
            src.update_ghost_values();
            dst.compress(VectorOperation::add);
            src.zero_out_ghost_values();
          }
        return;
      }

    if (computation_on)
      {
        if (dst.has_ghost_elements())
          dst.zero_out_ghost_values();

        const auto src_local_size =
          src.locally_owned_size() +
          (src.get_partitioner() ? src.get_partitioner()->n_ghost_indices() :
                                   0u);
        const auto dst_local_size =
          dst.locally_owned_size() +
          (dst.get_partitioner() ? dst.get_partitioner()->n_ghost_indices() :
                                   0u);

        Portable::DeviceVector<number> src_device(src.get_values(),
                                                  src_local_size),
          dst_device(dst.get_values(), dst_local_size);

        Kokkos::deep_copy(exec, dst_device, number(0));

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const unsigned int n_cells = n_cells_per_color[color];
            if (n_cells > 0)
              {
                constexpr bool is_serial =
                  std::is_same<Kokkos::DefaultExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>::value;

                const unsigned int numBlocks =
                  is_serial ? 1u : numbers::invalid_unsigned_int;
                const unsigned int threadsPerBlock =
                  is_serial ? 1u : numbers::invalid_unsigned_int;

                BK3Opt::Parallel::KokkosKernelAsync<dim,
                                                    fe_degree + 1,
                                                    nq,
                                                    number>(
                  exec,
                  shape_values,
                  co_shape_gradients,
                  G_tensors[color],
                  src_device,
                  dst_device,
                  dirichlet_boundary_dofs_masks[color],
                  n_cells,
                  numBlocks,
                  threadsPerBlock,
                  use_coloring_per_color[color]);
              }
          }

        copy_constrained_values(src, dst, exec);
        exec.fence();
      }
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::assemble_constant_rhs(
    VectorType  &dst,
    const number rhs_value) const
  {
    AssertDimension(dst.size(), m());

    ExecutionSpace exec;
    dst = 0.;

    const auto dst_local_size =
      dst.locally_owned_size() +
      (dst.get_partitioner() ? dst.get_partitioner()->n_ghost_indices() : 0u);
    BK3Opt::Parallel::DeviceView<number> dst_device(dst.get_values(),
                                                    dst_local_size);

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
    nelmtPerBatch =
      std::min(nelmtPerBatch, std::max(1u, 256u / (nq * nq)));

    const std::size_t shmem_size =
      fixed_shmem +
      2 * SharedView::shmem_size(nelmtPerBatch * nq_total);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = n_cells_per_color[color];
        if (n_cells == 0)
          continue;

        const unsigned int total_batches =
          (n_cells + nelmtPerBatch - 1) / nelmtPerBatch;

        internal::ConstantRhsKernelFunctor<dim, nm, nq, number> functor{
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
          std::is_same<Kokkos::DefaultExecutionSpace,
                       Kokkos::DefaultHostExecutionSpace>::value;

        unsigned int numBlocks       = is_serial ? 1u : total_batches;
        unsigned int threadsPerBlock = is_serial ? 1u : (nq * nq * nelmtPerBatch);

        TeamPolicy         probe_policy(exec, 1, Kokkos::AUTO);
        const unsigned int max_team_size = static_cast<unsigned int>(
          probe_policy.team_size_max(functor, Kokkos::ParallelForTag()));
        threadsPerBlock =
          std::max(1u, std::min(threadsPerBlock, max_team_size));

        TeamPolicy policy(exec, numBlocks, threadsPerBlock);
        Kokkos::parallel_for(
          "LaplaceOperatorOpt::assemble_constant_rhs", policy, functor);
      }

    if (has_ghost_exchange)
      dst.compress(VectorOperation::add);
    exec.fence();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::Tvmult(
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
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::initialize_dof_vector(
    VectorType &vec) const
  {
    if (partitioner)
      vec.reinit(partitioner);
    else
      vec.reinit(total_dofs);
  }

  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::m() const
  {
    return partitioner ? partitioner->size() : total_dofs;
  }

  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::n() const
  {
    return partitioner ? partitioner->size() : total_dofs;
  }

  template <int dim, int fe_degree, int nq, typename number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::get_vector_partitioner() const
  {
    return partitioner;
  }

  template <int dim, int fe_degree, int nq, typename number>
  BK3Opt::Parallel::ConstDeviceView<number>
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::get_G_tensor(
    const unsigned int color) const
  {
    AssertIndexRange(color, G_tensors.size());
    return G_tensors[color];
  }

  template <int dim, int fe_degree, int nq, typename number>
  std::size_t
  LaplaceOperatorOpt<dim, fe_degree, nq, number>::memory_consumption() const
  {
    std::size_t bytes =
      (shape_values.size() + co_shape_gradients.size() +
       quad_weights_1d.size()) *
        sizeof(number) +
      owned_constrained_dofs.size() * sizeof(unsigned int);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        bytes +=
          dirichlet_boundary_dofs_masks[color].size() * sizeof(unsigned int) +
          G_tensors[color].size() * sizeof(number);
      }

    return bytes;
  }

} // namespace PortableOpt

DEAL_II_NAMESPACE_CLOSE

#endif
