
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>


template <int dim, typename Number>
void
test_bk(const unsigned int degree,
        const unsigned int n_elements_in,
        const unsigned int n_tests,
        const bool         print_header)
{
  using namespace dealii;

  constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
  const unsigned int     n_mpi_ranks =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  const unsigned int n_elements =
    (n_elements_in / n_mpi_ranks + n_lanes - 1) / n_lanes * n_lanes;

  // create grid
  Triangulation<dim> tria;
  {
    const double              length = n_elements;
    std::vector<unsigned int> refinements(dim, 1);
    refinements[0] = n_elements;
    Point<dim> right;
    right[0] = length;
    for (unsigned int d = 1; d < dim; ++d)
      right[d] = 1.;

    GridGenerator::subdivided_hyper_rectangle(tria,
                                              refinements,
                                              Point<dim>(),
                                              right);

    // shift points on lower left boundary a bit to avoid Cartesian/affine cell
    // optimization of deal.II
    for (const auto &cell : tria.active_cell_iterators())
      cell->vertex(0)[1] =
        0.2 * std::sin(2. * cell->vertex(0)[0] * numbers::PI / length);
  }

  // create data structures for deal.II evaluation
  FE_DGQ<dim>     fe(degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraints;
  constraints.close();
  MappingQ1<dim> mapping;

  MatrixFree<dim, Number> matrix_free;
  matrix_free.reinit(
    mapping, dof_handler, constraints, QGauss<1>(degree + 2), {});

  AssertThrow(dof_handler.n_dofs() == n_elements * fe.dofs_per_cell,
              ExcDimensionMismatch(dof_handler.n_dofs(),
                                   n_elements * fe.dofs_per_cell));
  AlignedVector<Number> in(n_elements * fe.dofs_per_cell), out(in);
  for (unsigned int i = 0; i < in.size(); ++i)
    in[i] = 0.23 + 0.12 * std::sin(numbers::PI * i / (in.size() - 1)) -
            0.02 * std::sin(52. * numbers::PI * i / (in.size() - 1));

  AssertThrow(matrix_free.n_cell_batches() == n_elements / n_lanes,
              ExcDimensionMismatch(matrix_free.n_cell_batches(),
                                   n_elements / n_lanes));

  double best = 1e10, avg = 0, worst = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
#ifdef DEAL_II_WITH_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      Timer                               time;
      FEEvaluation<dim, -1, 0, 1, Number> eval(matrix_free);
      std::array<unsigned int, n_lanes>   dof_index_offsets;
      for (unsigned int l = 0; l < n_lanes; ++l)
        dof_index_offsets[l] = l * fe.dofs_per_cell;

      for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
        {
          eval.reinit(cell);
          vectorized_load_and_transpose(fe.dofs_per_cell,
                                        in.data() +
                                          cell * n_lanes * fe.dofs_per_cell,
                                        dof_index_offsets.data(),
                                        eval.begin_dof_values());

          eval.evaluate(EvaluationFlags::gradients);
          for (const unsigned int q : eval.quadrature_point_indices())
            eval.submit_gradient(eval.get_gradient(q), q);

          eval.integrate(EvaluationFlags::gradients);

          vectorized_transpose_and_store(false,
                                         fe.dofs_per_cell,
                                         eval.begin_dof_values(),
                                         dof_index_offsets.data(),
                                         out.data() +
                                           cell * n_lanes * fe.dofs_per_cell);
        }

      const double runtime = time.wall_time();
      best                 = std::min(best, runtime);
      worst                = std::max(worst, runtime);
      avg += runtime;
    }
  avg /= n_tests;

  const double throughput =
    Utilities::MPI::sum(1e-9 * dof_handler.n_dofs() / best, MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      if (print_header)
        std::cout << "test in "
                  << (std::is_same_v<Number, double> ? "FP64" : "FP32")
                  << std::endl
                  << "  p  |  q  |     n_dofs |    min_t |    avg_t |"
                  << "    max_t |   GDoF/s" << std::endl;
      std::cout << " " << std::setw(2) << degree << "  | " << std::setw(2)
                << degree + 2 << "  | " << std::setw(10) << dof_handler.n_dofs()
                << " | " << std::setw(8) << std::scientific
                << std::setprecision(2) << best << " | " << std::setw(8)
                << std::scientific << avg << " | " << std::setw(8)
                << std::scientific << worst << " | " << std::setw(8)
                << std::setprecision(3) << std::defaultfloat << throughput << std::endl;
    }
}



int
main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int size = 1000; size < 1000000; size *= 2)
    test_bk<3, double>(3, size, 10, size == 1000);
  for (unsigned int size = 1000; size < 1000000; size *= 2)
    test_bk<3, float>(3, size, 10, size == 1000);
  for (unsigned int size = 1000; size < 1000000; size *= 2)
    test_bk<3, float>(4, size, 10, size == 1000);
  for (unsigned int size = 1000; size < 1000000; size *= 2)
    test_bk<3, float>(5, size, 10, size == 1000);
}
