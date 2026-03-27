#ifndef portable_laplace_operator_h
#define portable_laplace_operator_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping_q1.h>

#include <memory>

#include "bk3_kokkos_kernel.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  
  template <int dim, int fe_degree, int nq, typename number>
  class LaplaceOperator
  {
  public:
    LaplaceOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<number> &constraints,
                    bool overlap_communication_computation);

    void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const ;

    void
    vmult_dummy(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
                &src,
      const bool ghost_exchange_on,
      const bool computation_on) const;

    void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const ;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
      const ;

    void
    setup_dirichlet_boundary_dofs_masks();

    types::global_dof_index
    m() const ;

    types::global_dof_index
    n() const ;

    const MatrixFree<dim, number> &
    get_matrix_free() const ;

    const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const ;

    void
    compute_G_tensors();

  private:
    using ExecutionSpace =
      dealii::MemorySpace::Default::kokkos_space::execution_space;
    using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
    using ViewValues = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ViewGradients = Kokkos::View<
      number **,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);

    static const unsigned int n_q_points = Utilities::pow(nq, dim);

    MatrixFree<dim, number> matrix_free;

    ObserverPointer<const AffineConstraints<number>> constraints;

    std::vector<
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      dirichlet_boundary_dofs_masks;

    std::vector<Kokkos::View<number *, MemorySpace::Default::kokkos_space>>
      G_tensors;
  };

  template <int dim, int fe_degree, int nq, typename number>
  LaplaceOperator<dim, fe_degree, nq, number>::LaplaceOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    bool                             overlap_communication_computation)
  {
    const MappingQ<dim> mapping(fe_degree);

    typename MatrixFree<dim, number>::AdditionalData additional_data;

    this->constraints = &constraints;

    additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    additional_data.overlap_communication_computation =
      overlap_communication_computation;

    const QGauss<1> quadrature_1d(nq);
    matrix_free.reinit(
      mapping, dof_handler, constraints, quadrature_1d, additional_data);

    setup_dirichlet_boundary_dofs_masks();

    compute_G_tensors();
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::vmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    dst = 0.;

    src.update_ghost_values();

    DeviceVector<number> src_device(src.get_values(), src.locally_owned_size()),
      dst_device(dst.get_values(), dst.locally_owned_size());

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        const unsigned int n_cells = colored_graph[color].size();

        if (n_cells > 0)
          {
            const auto &precomputed_data = matrix_free.get_data(color);

            unsigned int numBlocks       = numbers::invalid_unsigned_int;
            unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

            Kokkos::fence();

            BK3::Parallel::
              KokkosKernel<dim, fe_degree + 1, nq, number>(
                precomputed_data.shape_values,
                precomputed_data.co_shape_gradients,
                G_tensors[color],
                src_device,
                dst_device,
                dirichlet_boundary_dofs_masks[color],
                n_cells,
                numBlocks,
                threadsPerBlock);
            Kokkos::fence();
          }
      }

    dst.compress(VectorOperation::add);
    src.zero_out_ghost_values();
    matrix_free.copy_constrained_values(src, dst);
  }


  template <int dim, int fe_degree,int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::vmult_dummy(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    const bool ghost_exchange_on,
    const bool computation_on) const
  {


    if (ghost_exchange_on)
      src.update_ghost_values();

    if (computation_on)
      {

    dst = 0.;

        DeviceVector<number> src_device(src.get_values(),
                                        src.locally_owned_size()),
          dst_device(dst.get_values(), dst.locally_owned_size());

        const auto        &colored_graph = matrix_free.get_colored_graph();
        const unsigned int n_colors      = colored_graph.size();

        for (unsigned int color = 0; color < n_colors; ++color)
          {
            const unsigned int n_cells = colored_graph[color].size();

            if (n_cells > 0)
              {
                const auto &precomputed_data = matrix_free.get_data(color);

                unsigned int numBlocks       = numbers::invalid_unsigned_int;
                unsigned int threadsPerBlock = numbers::invalid_unsigned_int;

                Kokkos::fence();

                BK3::Parallel::
                  KokkosKernel<dim, fe_degree + 1, nq, number>(
                    precomputed_data.shape_values,
                    precomputed_data.co_shape_gradients,
                    G_tensors[color],
                    src_device,
                    dst_device,
                    dirichlet_boundary_dofs_masks[color],
                    n_cells,
                    numBlocks,
                    threadsPerBlock);
                Kokkos::fence();
              }
          }
      }

    if (ghost_exchange_on)
      {
        dst.compress(VectorOperation::add);
        src.zero_out_ghost_values();
        matrix_free.copy_constrained_values(src, dst);
      }
  }



  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::compute_G_tensors()
  {
    AssertDimension(dim, 3);

    constexpr int symmetric_tensor_dim = (dim * (dim + 1)) / 2;

    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    G_tensors.resize(n_colors);

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto        &precomputed_data = matrix_free.get_data(color);
            const unsigned int n_cells          = precomputed_data.n_cells;

            const auto &inv_jacobian = precomputed_data.inv_jacobian;
            const auto &JxW          = precomputed_data.JxW;

            G_tensors[color] =
              Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("G_tensor_color_" + std::to_string(color),
                                   Kokkos::WithoutInitializing),
                symmetric_tensor_dim * n_cells * n_q_points);

            auto G = G_tensors[color];

            Kokkos::parallel_for(
              "Fill_G_tensor_color" + std::to_string(color),
              Kokkos::RangePolicy<
                dealii::MemorySpace::Default::kokkos_space::execution_space>(
                0, n_cells),
              KOKKOS_LAMBDA(const int cell_id) {
                for (unsigned int q_point = 0; q_point < n_q_points; q_point++)
                  {
                    number components[symmetric_tensor_dim];

                    int idx = 0;
                    for (int d1 = 0; d1 < dim; ++d1)
                      for (int d2 = d1; d2 < dim; ++d2)
                        {
                          number sum = 0;
                          for (int k = 0; k < dim; ++k)
                            sum += inv_jacobian(q_point, cell_id, k, d1) *
                                   inv_jacobian(q_point, cell_id, k, d2);
                          components[idx] = JxW(q_point, cell_id) * sum;
                          ++idx;
                        }

                    for (int c = 0; c < symmetric_tensor_dim; ++c)
                      {
                        G[cell_id * symmetric_tensor_dim * n_q_points +
                          c * n_q_points + q_point] = components[c];
                      }
                  }
              });
            Kokkos::fence();
          }
      }
  }

  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::setup_dirichlet_boundary_dofs_masks()
  {
    dealii::MemorySpace::Default::kokkos_space::execution_space exec_space;
    const auto        &colored_graph = matrix_free.get_colored_graph();
    const unsigned int n_colors      = colored_graph.size();

    const auto &dof_handler = matrix_free.get_dof_handler();

    std::vector<unsigned int> lex_numbering(n_local_dofs);

    {
      const Quadrature<1> dummy_quadrature(
        std::vector<Point<1>>(1, Point<1>()));
      dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;


      shape_info.reinit(dummy_quadrature, dof_handler.get_fe(), 0);
      lex_numbering = shape_info.lexicographic_numbering;
    }

    this->dirichlet_boundary_dofs_masks.clear();
    this->dirichlet_boundary_dofs_masks.resize(n_colors);

    std::vector<types::global_dof_index> local_dof_indices(n_local_dofs);
    std::vector<types::global_dof_index> lexicographic_dof_indices(
      n_local_dofs);
    std::vector<types::global_dof_index> subdomain_local_dof_indices(
      n_local_dofs);

    const auto &partitioner = matrix_free.get_vector_partitioner();

    for (unsigned int color = 0; color < n_colors; ++color)
      {
        if (colored_graph[color].size() > 0)
          {
            const auto &mf_data = matrix_free.get_data(color);

            const auto &graph = colored_graph[color];

            this->dirichlet_boundary_dofs_masks[color] =
              Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
                Kokkos::view_alloc("dirichlet_boundary_dofs_" +
                                     std::to_string(color),
                                   Kokkos::WithoutInitializing),
                n_local_dofs,
                mf_data.n_cells);

            auto boundary_dofs_mask_host = Kokkos::create_mirror_view(
              this->dirichlet_boundary_dofs_masks[color]);

            for (unsigned int cell_id = 0; cell_id < mf_data.n_cells; ++cell_id)
              {
                auto triacell = graph[cell_id];

                typename DoFHandler<dim>::cell_iterator cell(
                  &(dof_handler.get_triangulation()),
                  triacell->level(),
                  triacell->index(),
                  &dof_handler);

                cell->get_dof_indices(local_dof_indices);

                triacell->get_dof_indices(subdomain_local_dof_indices);

                if (partitioner)
                  for (auto &index : local_dof_indices)
                    index = partitioner->global_to_local(index);

                for (unsigned int i = 0; i < n_local_dofs; ++i)
                  {
                    const auto global_dof = local_dof_indices[lex_numbering[i]];
                    const auto subdomain_local_dof =
                      subdomain_local_dof_indices[lex_numbering[i]];

                    if (constraints->is_constrained(subdomain_local_dof))
                      boundary_dofs_mask_host(i, cell_id) =
                        numbers::invalid_unsigned_int;
                    else
                      boundary_dofs_mask_host(i, cell_id) = global_dof;
                  }
              }

            Kokkos::deep_copy(exec_space,
                              this->dirichlet_boundary_dofs_masks[color],
                              boundary_dofs_mask_host);
            Kokkos::fence();
          }
      }
  }


  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::Tvmult(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src)
    const
  {
    AssertDimension(dst.size(), src.size());
    Assert(dst.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));
    Assert(src.get_partitioner() == matrix_free.get_vector_partitioner(),
           ExcMessage("Vector is not correctly initialized."));

    vmult(dst, src);
  }



  template <int dim, int fe_degree, int nq, typename number>
  void
  LaplaceOperator<dim, fe_degree, nq, number>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  template <int dim, int fe_degree, int nq, typename number>
  const MatrixFree<dim, number> &
  LaplaceOperator<dim, fe_degree, nq, number>::get_matrix_free() const
  {
    return matrix_free;
  }


  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, nq, number>::m() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }

  template <int dim, int fe_degree, int nq, typename number>
  types::global_dof_index
  LaplaceOperator<dim, fe_degree, nq, number>::n() const
  {
    return matrix_free.get_vector_partitioner()->size();
  }


  template <int dim, int fe_degree, int nq, typename number>
  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  LaplaceOperator<dim, fe_degree, nq, number>::get_vector_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
