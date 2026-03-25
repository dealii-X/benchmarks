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

#include "portable_laplace_operator.h"

using namespace dealii;

// Here at the top of the file, we collect the main global settings. The
// degree can be passed as the first argument to the program, but due to the
// templates we need to precompile the respective programs. Here we specify
// a minimum and maximum degree we want to support. Degrees outside this
// range will not do any work.
const unsigned int dimension      = 3;
const unsigned int minimal_degree = 1;
const unsigned int maximal_degree = 8;


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem();

  void
  run(const std::size_t min_size, const std::size_t max_size);

private:
  void
  setup_dofs();

  void
  setup_matrix_free();

  void
  compute_rhs();

  void
  solve();

  void
  do_matvec();

  void
  matvec_ghost_timing();

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

  double setup_time;

  ConvergenceTable convergence_table;

  ConvergenceTable ghost_timing_table;

  ConditionalOStream pcout;
  ConditionalOStream time_details;
};

template <int dim, int fe_degree>
LaplaceProblem<dim, fe_degree>::LaplaceProblem()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout,
                 true &&
                   Utilities::MPI::this_mpi_process(mpi_communicator) == 0)

{}

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

  setup_time += time.wall_time();

  time_details << "     DoFs and constraint set up  (CPU/wall)"
               << time.cpu_time() << "s/" << time.wall_time() << 's'
               << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::setup_matrix_free()
{
  Kokkos::fence();
  Timer time;

  system_matrix =
    std::make_unique<Portable::LaplaceOperator<dim, fe_degree, double>>(
      dof_handler, constraints, overlap_communication_computation);

  system_matrix->initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  ghost_solution_host.reinit(locally_owned_dofs,
                             locally_relevant_dofs,
                             mpi_communicator);

  Kokkos::fence();

  setup_time += time.wall_time();
  time_details << "     Setup matrices   (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << 's' << std::endl;
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::compute_rhs()
{
  Kokkos::fence();

  Timer time;

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

  Kokkos::fence();
  setup_time += time.wall_time();

  time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/"
               << time.wall_time() << 's' << std::endl;
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::do_matvec()
{
  system_matrix->vmult(solution_device, system_rhs_device);
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::solve()
{
  Kokkos::fence();
  Timer time;

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

  pcout << std::endl
        << "        Memory stats [MB]: " << memory.min << " [p"
        << memory.min_index << "] " << memory.avg << " " << memory.max << " [p"
        << memory.max_index << "]" << std::endl
        << std::endl;

  double                          time_cg = 1e10;
  std::pair<unsigned int, double> cg_details;
  for (unsigned int i = 0; i < 10; ++i)
    {
      ReductionControl solver_control(1000000000, 1e-16, 1e-9);
      SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
        cg(solver_control);
      solution_device = 0;

      Kokkos::fence();
      time.restart();
      cg.solve(*system_matrix,
               solution_device,
               system_rhs_device,
               PreconditionIdentity());
      Kokkos::fence();
      time_cg = std::min(time.wall_time(), time_cg);

      cg_details.first = solver_control.last_step();
      cg_details.second =
        std::pow(solver_control.last_value() / solver_control.initial_value(),
                 1. / solver_control.last_step());

      pcout << "Time solve CG              " << time.wall_time() << "\n";
    }

  double best_mv = 1e10;
  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

      Kokkos::fence();
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        do_matvec();
      Kokkos::fence();

      Utilities::MPI::MinMaxAvg stat =
        Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

      best_mv = std::min(best_mv, stat.max);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index
                  << "] " << stat.avg << " " << stat.max << " [p"
                  << stat.max_index << "]"
                  << " DoFs/s: "
                  << dof_handler.n_dofs() / stat.max /
                       Utilities::MPI::n_mpi_processes(mpi_communicator)
                  << std::endl;
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Best timings for ndof = " << dof_handler.n_dofs() << "   mv "
              << best_mv << "    CG total " << time_cg << "   CG per iter. "
              << time_cg / cg_details.first << std::endl;


  convergence_table.add_value("cells", triangulation.n_global_active_cells());
  convergence_table.add_value("dofs", dof_handler.n_dofs());
  convergence_table.add_value("matvec", best_mv);
  convergence_table.add_value("CG_tot_time", time_cg);
  convergence_table.add_value("CG_time/iters", time_cg / cg_details.first);
  convergence_table.add_value("cg_its", cg_details.first);
  convergence_table.add_value("cg_reduction", cg_details.second);
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::matvec_ghost_timing()
{
  const bool ghost_exchange_on = true;
  const bool computation_on    = true;

  Timer time;

  double best_mv_both    = 1e10;
  double best_only_ghost = 1e10;
  double best_only_comp  = 1e10;


  for (unsigned int i = 0; i < 5; ++i)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;
      {
        Kokkos::fence();
        time.restart();

        for (unsigned int i = 0; i < n_mv; ++i)
          system_matrix->vmult_dummy(solution_device,
                                     system_rhs_device,
                                     ghost_exchange_on,
                                     computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_mv_both = std::min(best_mv_both, stat.max);
      }
      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          system_matrix->vmult_dummy(solution_device,
                                     system_rhs_device,
                                     ghost_exchange_on,
                                     !computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_ghost = std::min(best_only_ghost, stat.max);
      }

      {
        Kokkos::fence();
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          system_matrix->vmult_dummy(solution_device,
                                     system_rhs_device,
                                     !ghost_exchange_on,
                                     computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);

        best_only_comp = std::min(best_only_comp, stat.max);
      }
    }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Best timings for ndof = " << dof_handler.n_dofs()
              << "|  ghost & compute =  " << best_mv_both
              << "   ghost only      =  " << best_only_ghost
              << "   compute only    =  " << best_only_comp << std::endl;

  ghost_timing_table.add_value("cells", triangulation.n_global_active_cells());
  ghost_timing_table.add_value("dofs", dof_handler.n_dofs());
  ghost_timing_table.add_value("mv_ghost_and_compute", best_mv_both);
  ghost_timing_table.add_value("mv_compute_only", best_only_comp);
  ghost_timing_table.add_value("mv_ghost_only", best_only_ghost);
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
LaplaceProblem<dim, fe_degree>::run(const std::size_t min_size,
                                    const std::size_t max_size)
{
  pcout << "Testing " << fe.get_name() << std::endl;
  pcout << "No. of GPUs: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << std::endl;


  const unsigned int sizes[] = {1,   2,   3,   4,    5,    6,   7,   8,
                                10,  12,  14,  16,   20,   24,  28,  32,
                                40,  48,  56,  64,   80,   96,  112, 128,
                                160, 192, 224, 256,  320,  384, 448, 512,
                                640, 768, 896, 1024, 1280, 1536};


  for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int);
       ++cycle)
    {
      triangulation.clear();

      setup_time = 0.;

      pcout << "Cycle " << cycle << std::endl;

      std::size_t  projected_size = numbers::invalid_size_type;
      unsigned int n_refine       = 0;

      {
        n_refine                     = cycle / 3;
        const unsigned int remainder = cycle % 3;
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
        const unsigned int base_refine = (1 << n_refine);
        projected_size                 = 1;
        for (unsigned int d = 0; d < dim; ++d)
          projected_size *= base_refine * subdivisions[d] * fe_degree + 1;
        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  subdivisions,
                                                  p1,
                                                  p2);
      }


      if (projected_size < min_size)
        continue;

      if (projected_size > max_size)
        {
          pcout << "Projected size " << projected_size
                << " higher than max size, terminating." << std::endl;
          pcout << std::endl;
          break;
        }

      triangulation.refine_global(n_refine);


      setup_dofs();

      setup_matrix_free();

      compute_rhs();

      pcout << "Total setup time: " << setup_time << std::endl;

      pcout << std::endl;
      solve();
      pcout << std::endl;

      pcout << std::endl;
      matvec_ghost_timing();
      pcout << std::endl;

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          convergence_table.set_scientific("matvec", true);
          convergence_table.set_precision("matvec", 3);
          convergence_table.set_scientific("CG_tot_time", true);
          convergence_table.set_precision("CG_tot_time", 3);
          convergence_table.set_scientific("CG_time/iters", true);
          convergence_table.set_precision("CG_time/iters", 3);
          convergence_table.set_scientific("cg_reduction", true);
          convergence_table.set_precision("cg_reduction", 3);

          convergence_table.write_text(std::cout);

          std::cout << std::endl << std::endl;

          ghost_timing_table.set_scientific("mv_ghost_and_compute", true);
          ghost_timing_table.set_precision("mv_ghost_and_compute", 4);
          ghost_timing_table.set_scientific("mv_compute_only", true);
          ghost_timing_table.set_precision("mv_compute_only", 4);
          ghost_timing_table.set_scientific("mv_ghost_only", true);
          ghost_timing_table.set_precision("mv_ghost_only", 4);

          ghost_timing_table.write_text(std::cout);

          std::cout << std::endl << std::endl;
        }
    }
}

template <int dim, int min_degree, int max_degree>
class LaplaceRunTime
{
public:
  LaplaceRunTime(const unsigned int target_degree,
                 const std::size_t  min_size,
                 const std::size_t  max_size)
  {
    if (min_degree > max_degree)
      return;
    if (min_degree == target_degree)
      {
        LaplaceProblem<dim, min_degree> laplace_problem;
        laplace_problem.run(min_size, max_size);
      }
    LaplaceRunTime<dim,
                   (min_degree <= max_degree ? (min_degree + 1) : min_degree),
                   max_degree>
      m(target_degree, min_size, max_size);
  }
};

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree  = numbers::invalid_unsigned_int;
      std::size_t  maxsize = static_cast<std::size_t>(-1);
      std::size_t  minsize = 1;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./timing_bp_5_3 degree minsize maxsize" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                  << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(degree,
                                                                    minsize,
                                                                    maxsize);
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