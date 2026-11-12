#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bk3_kokkos_kernel_opt.h"
#include "portable_cg_solver_opt2.h"
#include "portable_laplace_operator_opt2.h"

using namespace dealii;

const unsigned int dimension      = 3;
const unsigned int minimal_degree = 1;
const unsigned int maximal_degree = 8;

template <int dim, int fe_degree, int nq>
class LaplaceProblemOpt2
{
public:
  using VectorType =
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>;

  LaplaceProblemOpt2(const bool                 overlap_comm_comp = false,
                     const PortableOpt2::CGMode cg_mode =
                       PortableOpt2::CGMode::FusedExact);

  void
  run(const std::size_t min_size, const std::size_t max_size);

private:
  void
  release_cycle_memory();

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

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;

  IndexSet                                            locally_owned_dofs;
  IndexSet                                            locally_relevant_dofs;
  std::shared_ptr<const Utilities::MPI::Partitioner>  dof_partitioner;

  std::unique_ptr<PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, double>>
    system_matrix;

  GrowingVectorMemory<VectorType> solver_memory;

  VectorType solution_device;
  VectorType system_rhs_device;
  VectorType matvec_dst_device;

  const bool                 overlap_communication_computation;
  const PortableOpt2::CGMode cg_solver_mode;

  double setup_time;

  ConvergenceTable convergence_table;
  ConvergenceTable ghost_timing_table;

  ConditionalOStream pcout;
  ConditionalOStream time_details;
};

template <int dim, int fe_degree, int nq>
LaplaceProblemOpt2<dim, fe_degree, nq>::LaplaceProblemOpt2(
  const bool                 overlap_comm_comp,
  const PortableOpt2::CGMode cg_mode)
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator)
  , fe(fe_degree)
  , dof_handler(triangulation)
  , overlap_communication_computation(overlap_comm_comp)
  , cg_solver_mode(cg_mode)
  , setup_time(0.)
  , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  , time_details(std::cout,
                 Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
{}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::release_cycle_memory()
{
  // Release pooled CG scratch vectors and move-reset cycle vectors without
  // triggering deal.II reinit(0) stub allocations or skipped-cycle churn.
  solver_memory.release_unused_memory();
  matvec_dst_device = VectorType();
  system_rhs_device = VectorType();
  solution_device   = VectorType();
  system_matrix.reset();
  dof_partitioner.reset();
  Kokkos::fence();
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::setup_dofs()
{
  Timer time;

  dof_handler.reinit(triangulation);
  dof_handler.distribute_dofs(fe);

  pcout << "  Number of cells: " << triangulation.n_global_active_cells()
        << " | "
        << "  Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Build MPI Partitioner ONCE here and share it directly with LaplaceOperatorOpt2
  dof_partitioner = std::make_shared<Utilities::MPI::Partitioner>(
    locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  const double wall = time.wall_time();
  setup_time += wall;

  time_details << "     DoFs and constraint set up  (CPU/wall)"
               << time.cpu_time() << "s/" << wall << 's' << std::endl;
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::setup_matrix_free()
{
  Kokkos::fence();
  Timer time;

  system_matrix = std::make_unique<
    PortableOpt2::LaplaceOperatorOpt2<dim, fe_degree, nq, double>>(
    dof_handler,
    constraints,
    overlap_communication_computation,
    dof_partitioner);

  system_matrix->initialize_dof_vector(solution_device);
  system_rhs_device.reinit(solution_device);
  matvec_dst_device.reinit(solution_device);

  Kokkos::fence();

  const double wall = time.wall_time();
  setup_time += wall;
  time_details << "     Setup matrices   (CPU/wall) " << time.cpu_time() << "s/"
               << wall << 's' << std::endl;
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::compute_rhs()
{
  Kokkos::fence();

  Timer time;

  system_matrix->assemble_constant_rhs(system_rhs_device, 1.0);

  Kokkos::fence();
  const double wall = time.wall_time();
  setup_time += wall;

  time_details << "Compute rhs   (CPU/wall) " << time.cpu_time() << "s/" << wall
               << 's' << std::endl;
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::do_matvec()
{
  system_matrix->vmult(matvec_dst_device, system_rhs_device);
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::solve()
{
  Kokkos::fence();
  Timer time;

  Utilities::System::MemoryStats stats;
  Utilities::System::get_memory_stats(stats);
  Utilities::MPI::MinMaxAvg memory =
    Utilities::MPI::min_max_avg(stats.VmRSS / 1024., mpi_communicator);

  pcout << std::endl
        << "        Memory stats [MB]: " << memory.min << " [p"
        << memory.min_index << "] " << memory.avg << " " << memory.max << " [p"
        << memory.max_index << "]" << std::endl
        << std::endl;

  // Pre-warm the 3 CG scratch vectors in solver_memory BEFORE the timed loop
  // so timed CG solves measure pure solver execution rather than cudaMalloc/Free
  {
    typename VectorMemory<VectorType>::Pointer v1(solver_memory);
    typename VectorMemory<VectorType>::Pointer v2(solver_memory);
    typename VectorMemory<VectorType>::Pointer v3(solver_memory);
    v1->reinit(solution_device, /*omit_zeroing_entries=*/true);
    v2->reinit(solution_device, /*omit_zeroing_entries=*/true);
    v3->reinit(solution_device, /*omit_zeroing_entries=*/true);
  }

  const typename PortableOpt2::SolverCGOpt2<VectorType>::AdditionalData cg_data(
    cg_solver_mode);

  double                          time_cg = 1e10;
  std::pair<unsigned int, double> cg_details;
  for (unsigned int i = 0; i < 10; ++i)
    {
      ReductionControl               solver_control(1000000000, 1e-16, 1e-9);
      PortableOpt2::SolverCGOpt2<VectorType> cg(solver_control,
                                                solver_memory,
                                                cg_data);
      solution_device = 0.;

      Kokkos::fence();
      time.restart();
      cg.solve(*system_matrix,
               solution_device,
               system_rhs_device,
               PreconditionIdentity());
      Kokkos::fence();

      const Utilities::MPI::MinMaxAvg cg_stat =
        Utilities::MPI::min_max_avg(time.wall_time(), mpi_communicator);
      time_cg = std::min(cg_stat.max, time_cg);

      cg_details.first = solver_control.last_step();
      cg_details.second =
        (solver_control.last_step() > 0) ?
          std::pow(solver_control.last_value() / solver_control.initial_value(),
                   1. / solver_control.last_step()) :
          0.0;

      pcout << "Time solve CG              " << cg_stat.max << "\n";
    }

  double best_mv = 1e10;
  for (unsigned int rep = 0; rep < 5; ++rep)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;

      Kokkos::fence();
      time.restart();
      for (unsigned int j = 0; j < n_mv; ++j)
        do_matvec();
      Kokkos::fence();

      Utilities::MPI::MinMaxAvg stat =
        Utilities::MPI::min_max_avg(time.wall_time() / n_mv, mpi_communicator);

      best_mv = std::min(best_mv, stat.max);

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index
                  << "] " << stat.avg << " " << stat.max << " [p"
                  << stat.max_index << "]"
                  << " DoFs/s: "
                  << dof_handler.n_dofs() / stat.max /
                       Utilities::MPI::n_mpi_processes(mpi_communicator)
                  << std::endl;
    }

  const double cg_per_iter =
    (cg_details.first > 0) ? (time_cg / cg_details.first) : 0.0;

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::cout << "Best timings for ndof = " << dof_handler.n_dofs() << "   mv "
              << best_mv << "    CG total " << time_cg << "   CG per iter. "
              << cg_per_iter << std::endl;

  convergence_table.add_value("cells", triangulation.n_global_active_cells());
  convergence_table.add_value("dofs", dof_handler.n_dofs());
  convergence_table.add_value("matvec", best_mv);
  convergence_table.add_value("CG_tot_time", time_cg);
  convergence_table.add_value("CG_time/iters", cg_per_iter);
  convergence_table.add_value("cg_its", cg_details.first);
  convergence_table.add_value("cg_reduction", cg_details.second);
}

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::matvec_ghost_timing()
{
  const bool ghost_exchange_on = true;
  const bool computation_on    = true;

  Timer time;

  double best_mv_both    = 1e10;
  double best_only_ghost = 1e10;
  double best_only_comp  = 1e10;

  for (unsigned int rep = 0; rep < 5; ++rep)
    {
      const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;
      {
        Kokkos::fence();
        time.restart();

        for (unsigned int j = 0; j < n_mv; ++j)
          system_matrix->vmult_dummy(matvec_dst_device,
                                     system_rhs_device,
                                     ghost_exchange_on,
                                     computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv,
                                      mpi_communicator);

        best_mv_both = std::min(best_mv_both, stat.max);
      }
      {
        Kokkos::fence();
        time.restart();
        for (unsigned int j = 0; j < n_mv; ++j)
          system_matrix->vmult_dummy(matvec_dst_device,
                                     system_rhs_device,
                                     ghost_exchange_on,
                                     !computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv,
                                      mpi_communicator);

        best_only_ghost = std::min(best_only_ghost, stat.max);
      }
      {
        Kokkos::fence();
        time.restart();
        for (unsigned int j = 0; j < n_mv; ++j)
          system_matrix->vmult_dummy(matvec_dst_device,
                                     system_rhs_device,
                                     !ghost_exchange_on,
                                     computation_on);
        Kokkos::fence();

        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv,
                                      mpi_communicator);

        best_only_comp = std::min(best_only_comp, stat.max);
      }
    }

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
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

template <int dim, int fe_degree, int nq>
void
LaplaceProblemOpt2<dim, fe_degree, nq>::run(const std::size_t min_size,
                                            const std::size_t max_size)
{
  pcout << "Testing " << fe.get_name() << std::endl;
  pcout << "No. of GPUs: " << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << std::endl;

  constexpr unsigned int max_cycles = 38;

  for (unsigned int cycle = 0; cycle < max_cycles; ++cycle)
    {
      const unsigned int        n_refine  = cycle / 3;
      const unsigned int        remainder = cycle % 3;
      std::vector<unsigned int> subdivisions(dim, 1);
      for (unsigned int d = 0; d < remainder; ++d)
        subdivisions[d] = 2;

      const std::size_t base_refine    = (std::size_t(1) << n_refine);
      std::size_t       projected_size = 1;
      for (unsigned int d = 0; d < dim; ++d)
        projected_size *= base_refine * subdivisions[d] * fe_degree + 1;

      if (projected_size < min_size)
        continue;

      if (projected_size > max_size)
        {
          pcout << "Cycle " << cycle << std::endl
                << "Projected size " << projected_size
                << " higher than max size, terminating." << std::endl
                << std::endl;
          break;
        }

      release_cycle_memory();
      triangulation.clear();

      setup_time = 0.;

      pcout << "Cycle " << cycle << std::endl;

      Point<dim> p1;
      for (unsigned int d = 0; d < dim; ++d)
        p1[d] = -1.0;
      Point<dim> p2;
      for (unsigned int d = 0; d < remainder; ++d)
        p2[d] = 2.8;
      for (unsigned int d = remainder; d < dim; ++d)
        p2[d] = 0.9;

      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                subdivisions,
                                                p1,
                                                p2);
      triangulation.refine_global(n_refine);

      setup_dofs();
      setup_matrix_free();
      compute_rhs();

      pcout << "Total setup time: " << setup_time << std::endl << std::endl;
      solve();
      pcout << std::endl << std::endl;
      matvec_ghost_timing();
      pcout << std::endl;

      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
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

  release_cycle_memory();
}

template <int dim, int min_degree, int max_degree>
class LaplaceRunTimeOpt2
{
public:
  LaplaceRunTimeOpt2(const unsigned int         target_degree,
                     const std::size_t          min_size,
                     const std::size_t          max_size,
                     const bool                 overlap_comm_comp,
                     const PortableOpt2::CGMode cg_mode)
  {
    if constexpr (min_degree <= max_degree)
      {
        if (min_degree == target_degree)
          {
            constexpr int nq = min_degree + 2;
            LaplaceProblemOpt2<dim, min_degree, nq> laplace_problem(
              overlap_comm_comp, cg_mode);
            laplace_problem.run(min_size, max_size);
            return;
          }
        if constexpr (min_degree < max_degree)
          {
            LaplaceRunTimeOpt2<dim, min_degree + 1, max_degree> next(
              target_degree, min_size, max_size, overlap_comm_comp, cg_mode);
          }
      }
  }
};

namespace
{
  std::size_t
  parse_positive_ull(const std::string &s, const char *name)
  {
    if (s.empty() || s[0] == '-')
      throw std::invalid_argument(std::string("Invalid negative or empty ") +
                                  name + ": '" + s + "'");
    std::size_t       pos = 0;
    const std::size_t val = std::stoull(s, &pos);
    if (pos != s.size())
      throw std::invalid_argument(std::string("Invalid numeric argument for ") +
                                  name + ": '" + s + "'");
    return val;
  }

  const char *
  cg_mode_name(const PortableOpt2::CGMode mode)
  {
    switch (mode)
      {
        case PortableOpt2::CGMode::FusedExact:
          return "FusedExact (2 syncs/iter, exact CG iterates)";
        case PortableOpt2::CGMode::FusedSingleAllreduce:
          return "FusedSingleAllreduce (1 sync + 1 Allreduce/iter)";
        case PortableOpt2::CGMode::DealII:
          return "dealii::SolverCG";
      }
    return "Unknown";
  }
} // namespace

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      PortableOpt2::CGMode     cg_mode          = PortableOpt2::CGMode::FusedExact;
      bool                     overlap_override = false;
      bool                     overlap_flag_val = false;
      std::vector<std::string> positional;

      for (int i = 1; i < argc; ++i)
        {
          const std::string arg(argv[i]);
          if (arg == "--dealii-cg")
            cg_mode = PortableOpt2::CGMode::DealII;
          else if (arg == "--single-allreduce-cg" || arg == "--pipelined-cg")
            cg_mode = PortableOpt2::CGMode::FusedSingleAllreduce;
          else if (arg == "--fused-cg")
            cg_mode = PortableOpt2::CGMode::FusedExact;
          else if (arg == "--no-overlap")
            {
              overlap_override = true;
              overlap_flag_val = false;
            }
          else if (arg == "--overlap")
            {
              overlap_override = true;
              overlap_flag_val = true;
            }
          else
            positional.push_back(arg);
        }

      if (positional.empty())
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "Expected at least one argument." << std::endl
              << "Usage:" << std::endl
              << "./bp3_opt2 <degree> [minsize] [maxsize] [overlap=0|1] "
                 "[--fused-cg|--single-allreduce-cg|--dealii-cg] "
                 "[--overlap|--no-overlap]"
              << std::endl;
          return 1;
        }

      const std::size_t parsed_degree =
        parse_positive_ull(positional[0], "degree");
      if (parsed_degree < minimal_degree || parsed_degree > maximal_degree)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cerr << "Error: Polynomial degree must be between "
                      << minimal_degree << " and " << maximal_degree
                      << " (got " << parsed_degree << ")." << std::endl;
          return 1;
        }

      const unsigned int degree = static_cast<unsigned int>(parsed_degree);
      const std::size_t  minsize =
        (positional.size() > 1) ? parse_positive_ull(positional[1], "minsize") :
                                  1;
      const std::size_t maxsize =
        (positional.size() > 2) ? parse_positive_ull(positional[2], "maxsize") :
                                  std::numeric_limits<std::size_t>::max();
      bool overlap =
        (positional.size() > 3) ?
          (parse_positive_ull(positional[3], "overlap") != 0) :
          (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1);
      if (overlap_override)
        overlap = overlap_flag_val;

      if (minsize > maxsize)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cerr << "Error: minsize (" << minsize
                      << ") cannot exceed maxsize (" << maxsize << ")."
                      << std::endl;
          return 1;
        }

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                  << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Overlap comm/computation:       "
                  << (overlap ? "true" : "false") << std::endl
                  << "GPU-aware MPI runtime path:     "
                  << (PortableOpt2::query_device_aware_mpi() ?
                        "direct device pointers" :
                        "pinned host DMA staging")
                  << std::endl
                  << "CG solver mode:                 " << cg_mode_name(cg_mode)
                  << std::endl
                  << std::endl;

      LaplaceRunTimeOpt2<dimension, minimal_degree, maximal_degree> run(
        degree, minsize, maxsize, overlap, cg_mode);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl
                << "Exception on processing: " << std::endl
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
                << std::endl
                << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
