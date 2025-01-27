/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2019 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Authors: Bruno Turcksin, Daniel Arndt, Oak Ridge National Laboratory, 2019
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <fstream>

#include "create_triangulation.h"


namespace Step64
{
  using namespace dealii;


  template <int dim, int fe_degree>
  class HelmholtzOperatorQuad
  {
  public:
    DEAL_II_HOST_DEVICE HelmholtzOperatorQuad(
      const typename Portable::MatrixFree<dim, double>::Data *gpu_data,
      int                                                     cell)
      : gpu_data(gpu_data)
      , cell(cell)
    {}

    DEAL_II_HOST_DEVICE void operator()(
      Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> *fe_eval,
      const int q_point) const;

    DEAL_II_HOST_DEVICE void set_matrix_free_data(
      const typename Portable::MatrixFree<dim, double>::Data &data)
    {
      gpu_data = &data;
    }

    DEAL_II_HOST_DEVICE void set_cell(int new_cell)
    {
      cell = new_cell;
    }

    static const unsigned int n_q_points =
      dealii::Utilities::pow(fe_degree + 1, dim);

    static const unsigned int n_local_dofs = n_q_points;

  private:
    const typename Portable::MatrixFree<dim, double>::Data *gpu_data;
    int                                                     cell;
  };


  template <int dim, int fe_degree>
  DEAL_II_HOST_DEVICE void HelmholtzOperatorQuad<dim, fe_degree>::operator()(
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> *fe_eval,
    const int q_point) const
  {
    fe_eval->submit_value(fe_eval->get_value(q_point), q_point);
    fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
  }



  template <int dim, int fe_degree>
  class LocalHelmholtzOperator
  {
  public:
    static constexpr unsigned int n_dofs_1d = fe_degree + 1;
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim);

    LocalHelmholtzOperator()
    {}

    DEAL_II_HOST_DEVICE void
    operator()(const unsigned int                                      cell,
               const typename Portable::MatrixFree<dim, double>::Data *gpu_data,
               Portable::SharedData<dim, double> *shared_data,
               const double                      *src,
               double                            *dst) const;
  };


  template <int dim, int fe_degree>
  DEAL_II_HOST_DEVICE void LocalHelmholtzOperator<dim, fe_degree>::operator()(
    const unsigned int                                      cell,
    const typename Portable::MatrixFree<dim, double>::Data *gpu_data,
    Portable::SharedData<dim, double>                      *shared_data,
    const double                                           *src,
    double                                                 *dst) const
  {
    Portable::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, double> fe_eval(
      gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
    fe_eval.apply_for_each_quad_point(
      HelmholtzOperatorQuad<dim, fe_degree>(gpu_data, cell));
    fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
    fe_eval.distribute_local_to_global(dst);
  }



  template <int dim, int fe_degree>
  class HelmholtzOperator : public EnableObserverPointer
  {
  public:
    HelmholtzOperator(const DoFHandler<dim>           &dof_handler,
                      const AffineConstraints<double> &constraints);

    void
    vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
            &src) const;

    void initialize_dof_vector(
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &vec)
      const;

    void compute_diagonal();

    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
    get_matrix_diagonal_inverse() const;

    types::global_dof_index m() const;

    types::global_dof_index n() const;

    double el(const types::global_dof_index row,
              const types::global_dof_index col) const;

  private:
    Portable::MatrixFree<dim, double>                                mf_data;
    std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
      inverse_diagonal_entries;
  };



  template <int dim, int fe_degree>
  HelmholtzOperator<dim, fe_degree>::HelmholtzOperator(
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<double> &constraints)
  {
    const MappingQ<dim> mapping(fe_degree);
    typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping, dof_handler, constraints, quad, additional_data);
  }



  template <int dim, int fe_degree>
  void HelmholtzOperator<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>       &dst,
    const LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &src)
    const
  {
    dst = 0.;
    LocalHelmholtzOperator<dim, fe_degree> helmholtz_operator;
    mf_data.cell_loop(helmholtz_operator, src, dst);
    mf_data.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree>
  void HelmholtzOperator<dim, fe_degree>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> &vec) const
  {
    mf_data.initialize_dof_vector(vec);
  }



  template <int dim, int fe_degree>
  void HelmholtzOperator<dim, fe_degree>::compute_diagonal()
  {
    this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<
        LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>());
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
      &inverse_diagonal = inverse_diagonal_entries->get_vector();
    initialize_dof_vector(inverse_diagonal);

    HelmholtzOperatorQuad<dim, fe_degree> helmholtz_operator_quad(
      nullptr, -1);

    MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 0, double>(
      mf_data,
      inverse_diagonal,
      helmholtz_operator_quad,
      EvaluationFlags::values | EvaluationFlags::gradients,
      EvaluationFlags::values | EvaluationFlags::gradients);

    double *raw_diagonal = inverse_diagonal.get_values();

    Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        /*Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        */
        if (raw_diagonal[i] > 0)
          raw_diagonal[i] = 1. / raw_diagonal[i];
        else
          raw_diagonal[i] = 1.;
      });
  }



  template <int dim, int fe_degree>
  std::shared_ptr<DiagonalMatrix<
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
  HelmholtzOperator<dim, fe_degree>::get_matrix_diagonal_inverse() const
  {
    return inverse_diagonal_entries;
  }



  template <int dim, int fe_degree>
  types::global_dof_index HelmholtzOperator<dim, fe_degree>::m() const
  {
    return mf_data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree>
  types::global_dof_index HelmholtzOperator<dim, fe_degree>::n() const
  {
    return mf_data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree>
  double
  HelmholtzOperator<dim, fe_degree>::el(const types::global_dof_index row,
                                        const types::global_dof_index col) const
  {
    (void)col;
    Assert(row == col, ExcNotImplemented());
    Assert(inverse_diagonal_entries.get() != nullptr &&
             inverse_diagonal_entries->m() > 0,
           ExcNotInitialized());
    return 1.0 / (*inverse_diagonal_entries)(row, row);
  }



  template <int dim, int fe_degree>
  void
  test(const unsigned int s,
       const bool         short_output)
{
  const unsigned int n_q_points = fe_degree + 1;

  Timer           time;
  const auto tria = create_triangulation<dim>(s, true);

  FE_Q<dim>            fe_q(fe_degree);
  MappingQ<dim> mapping(1);
  DoFHandler<dim>      dof_handler(*tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(dof_handler,
                                          0,
                                          Functions::ZeroFunction<dim>(),
                                          constraints);
  constraints.close();

  HelmholtzOperator<dim, fe_degree> helmholtz(dof_handler, constraints);

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
    ghost_solution_host;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    solution_dev;
  LinearAlgebra::distributed::Vector<double, MemorySpace::Default>
    system_rhs_dev;

  ghost_solution_host.reinit(dof_handler.locally_owned_dofs(),
                             relevant_dofs,
                             MPI_COMM_WORLD);
  helmholtz.initialize_dof_vector(solution_dev);
  system_rhs_dev.reinit(solution_dev);

  helmholtz.compute_diagonal();
  const std::shared_ptr<DiagonalMatrix<
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>>
    preconditioner = helmholtz.get_matrix_diagonal_inverse();

  for (unsigned int i = 0; i < ghost_solution_host.locally_owned_size(); ++i)
    if (!constraints.is_constrained(ghost_solution_host.get_partitioner()->local_to_global(i)))
      ghost_solution_host.local_element(i) = (i) % 8;

  LinearAlgebra::ReadWriteVector<double> rw_vector(dof_handler.locally_owned_dofs());
  rw_vector.import_elements(ghost_solution_host, VectorOperation::insert);
  system_rhs_dev.import_elements(rw_vector, VectorOperation::insert);

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  ReductionControl                                     solver_control(100, 1e-15, 1e-8);
  SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>> solver(solver_control);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver");
#endif
  double solver_time = 1e10;
  for (unsigned int t = 0; t < 4; ++t)
    {
      solution_dev = 0;
      time.restart();
      try
        {
          solver.solve(helmholtz, solution_dev, system_rhs_dev, *preconditioner);
        }
      catch (SolverControl::NoConvergence &e)
        {
          // prevent the solver to throw an exception in case we should need more
          // than 100 iterations
        }
      Kokkos::fence();
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      solver_time = std::min(data.max, solver_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver");
#endif

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec");
#endif
  double matvec_time = 1e10;
  for (unsigned int t = 0; t < 4; ++t)
    {
      time.restart();
      for (unsigned int i = 0; i < 50; ++i)
        helmholtz.vmult(system_rhs_dev, solution_dev);
      Kokkos::fence();
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max / 50, matvec_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec");
#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    std::cout << std::setw(2) << fe_degree << " | " << std::setw(2) << n_q_points   //
              << " |" << std::setw(10) << tria->n_global_active_cells()             //
              << " |" << std::setw(11) << dof_handler.n_dofs()                      //
              << " | " << std::setw(11) << solver_time / solver_control.last_step() //
              << " | " << std::setw(11)
              << dof_handler.n_dofs() / solver_time * solver_control.last_step()    //
              << " | " << std::setw(4) << solver_control.last_step()                 //
              << " | " << std::setw(11) << matvec_time                               //
              << std::endl;
}


  template <int dim, int fe_degree>
void
do_test(const int s_in, const bool compact_output)
{
  if (s_in < 1)
    {
      unsigned int s =
        std::max(3U,
                 static_cast<unsigned int>(std::log2(1024 / fe_degree / fe_degree / fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout
          << " p |  q | n_element |     n_dofs |     time/it |   dofs/s/it | itCG | time/matvec"
          << std::endl;
      while ((2 + Utilities::fixed_power<dim>(fe_degree + 1)) * (1UL << (s / 4)) <
             12000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          test<dim, fe_degree>(s, compact_output);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    test<dim, fe_degree> (s_in, compact_output);

}
}


int
main(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  unsigned int degree         = 1;
  unsigned int s              = -1;
  bool         compact_output = true;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    s = std::atoi(argv[2]);
  if (argc > 3)
    compact_output = std::atoi(argv[3]);

  if (degree == 1)
    Step64::do_test<3, 1>(s, compact_output);
  //else if (degree == 2)
  //  Step64::do_test<3, 2>(s, compact_output);
  //else if (degree == 3)
  //  Step64::do_test<3, 3>(s, compact_output);
  else if (degree == 4)
    Step64::do_test<3, 4>(s, compact_output);
  //else if (degree == 5)
  //  Step64::do_test<3, 5>(s, compact_output);
    //else if (degree == 6)
  // Step64::do_test<3, 6>(s, compact_output);
  //else if (degree == 7)
  //  Step64::do_test<3, 7>(s, compact_output);
  else
    {
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Degree " << degree << " not implemented" << std::endl;
    }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
