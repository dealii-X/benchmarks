#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <memory>

#include "bk3_kokkos_kernel.h"
#include "portable_laplace_operator.h"

using namespace dealii;


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem();

  void
  run();

private:
  void
  setup_grid(unsigned int refinement_cycles);

  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  compute_rhs();

  void
  solve();

  void
  postprocess_solution();

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;


  std::unique_ptr<Portable::LaplaceOperator<dim, fe_degree, double>>
    system_matrix;

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_device;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_device;

  const bool overlap_communication_computation = false;

  ConditionalOStream pcout;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)

{}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_grid(unsigned int refinement_cycles)
{
  triangulation.clear();

  unsigned int       n_refine  = refinement_cycles / 3;
  const unsigned int remainder = refinement_cycles % 3;
  Point<dim>         p1;
  for (unsigned int d = 0; d < dim; ++d)
    p1[d] = -1;
  Point<dim> p2;
  for (unsigned int d = 0; d < remainder; ++d)
    p2[d] = 2.8;
  for (unsigned int d = remainder; d < dim; ++d)
    p2[d] = 0.9;
  std::vector<unsigned int> subdivisions(dim, 1);
  for (unsigned int d = 0; d < remainder; ++d)
    subdivisions[d] = 2;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
                                            subdivisions,
                                            p1,
                                            p2);

  triangulation.refine_global(n_refine);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_dofs()
{
  Timer time;

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  pcout << "  Number of cells: " << triangulation.n_global_active_cells()
        << " | "
        << "  Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  system_matrix =
    std::make_unique<Portable::LaplaceOperator<dim, fe_degree, double>>(
      dof_handler, constraints, overlap_communication_computation);

  system_matrix->initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_rhs()
{
  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> system_rhs_host(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          cell_rhs = 0;

          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += (fe_values.shape_value(i, q_index) * 1.0 *
                              fe_values.JxW(q_index));

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_rhs,
                                                 local_dof_indices,
                                                 system_rhs_host);
        }
    }

  system_rhs_host.compress(VectorOperation::add);
  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);

  rw_vector.import_elements(system_rhs_host, VectorOperation::insert);
  system_rhs_device.import_elements(rw_vector, VectorOperation::insert);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve()
{
  ReductionControl solver_control(system_rhs_device.size(), 1e-12, 1e-6);

  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> cg(
    solver_control);

  solution_device = 0;

  cg.solve(*system_matrix,
           solution_device,
           system_rhs_device,
           PreconditionIdentity());

  pcout << "  Solved in " << solver_control.last_step() << " iterations."
        << std::endl;
}


template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::postprocess_solution()
{
  LinearAlgebra::ReadWriteVector<double> rw_vector(locally_owned_dofs);
  rw_vector.import_elements(solution_device, VectorOperation::insert);
  ghost_solution_host.import_elements(rw_vector, VectorOperation::insert);

  constraints.distribute(ghost_solution_host);

  ghost_solution_host.update_ghost_values();
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  // const unsigned int sizes[] = {1,   2,   3,   4,    5,    6,   7,   8,
  //                               10,  12,  14,  16,   20,   24,  28,  32,
  //                               40,  48,  56,  64,   80,   96,  112, 128,
  //                               160, 192, 224, 256,  320,  384, 448, 512,
  //                               640, 768, 896, 1024, 1280, 1536};

  const unsigned int sizes[] = {1, 2, 3, 4};

  for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int);
       ++cycle)
    {
      triangulation.clear();

      pcout << "Cycle " << cycle << std::endl;

      setup_grid(cycle);

      setup_dofs();

      setup_matrix_free();

      compute_rhs();

      solve();

      postprocess_solution();

      const auto &mf            = system_matrix->get_matrix_free();
      const auto &colored_graph = mf.get_colored_graph();

      if (colored_graph.size() > 0)
        {
          if (colored_graph[0].size() > 0)
            {
              const auto &gpu_data = mf.get_data(0);

              // std::cout << "inv_jacobian.size() =  "
              //           << gpu_data.inv_jacobian.size() << " | "
              //           << "JxW.size() =  " << gpu_data.JxW.size() <<
              //           std::endl;

              std::cout << "shape_values.size() =  "
                        << gpu_data.shape_values.size() << " | "
                        << "co_shape_gradients.size() =  "
                        << gpu_data.co_shape_gradients.size() << std::endl;
            }
        }
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      constexpr unsigned int dim       = 3;
      constexpr unsigned int fe_degree = 1;

      LaplaceProblem<dim, fe_degree> laplace_problem;
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}