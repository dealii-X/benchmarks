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

#include <deal.II/matrix_free/portable_fe_evaluation.h>
#include <deal.II/matrix_free/portable_matrix_free.h>
#include <deal.II/matrix_free/portable_matrix_free.templates.h>

#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>

#include "portable_laplace_operator.h"
#include "portable_laplace_operator_opt.h"

using namespace dealii;

namespace
{
  // Smooth, anisotropic, sheared, and non-affine 3D diffeomorphism with exact
  // analytical inverse. Exercises all 6 independent entries of G = JxW * J^{-1} J^{-T}
  // as well as curved high-order MappingQ(p) geometry.
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
    static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
    static constexpr unsigned int n_q_points = Utilities::pow(nq, dim);

    DEAL_II_HOST_DEVICE void
    operator()(const typename Portable::MatrixFree<dim, Number>::Data *data,
               const Portable::DeviceVector<Number>                   &src,
               Portable::DeviceVector<Number> &dst) const
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

  // Ground-truth matrix-free Laplace operator using deal.II's native
  // Portable::FEEvaluation.
  template <int dim, int fe_degree, int nq, typename Number>
  class DealIIReferenceLaplaceOperator
  {
  public:
    DealIIReferenceLaplaceOperator(const DoFHandler<dim>           &dof_handler,
                                   const AffineConstraints<Number> &constraints)
    {
      const MappingQ<dim> mapping(fe_degree);
      typename Portable::MatrixFree<dim, Number>::AdditionalData
        additional_data;
      additional_data.mapping_update_flags =
        update_gradients | update_JxW_values;
      additional_data.overlap_communication_computation = false;
      const QGauss<1> quad(nq);
      matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &vec)
      const
    {
      matrix_free.initialize_dof_vector(vec);
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
            &src) const
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
    }

  private:
    Portable::MatrixFree<dim, Number> matrix_free;
  };

  template <int dim, int fe_degree, typename Number>
  void
  run_deformed_operator_test(MPI_Comm           comm,
                             ConditionalOStream &pcout,
                             const double       rel_tol)
  {
    constexpr int nq = fe_degree + 2;

    parallel::distributed::Triangulation<dim> tria(comm);
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    GridTools::transform(&CurvedShearedManifold<dim>::map_forward, tria);

    CurvedShearedManifold<dim> manifold;
    tria.set_all_manifold_ids(0);
    tria.set_manifold(0, manifold);
    tria.refine_global(2); // 64 curved, sheared, anisotropic 3D cells

    FE_Q<dim>       fe(fe_degree);
    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

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
    PortableOpt::LaplaceOperatorOpt<dim, fe_degree, nq, Number> opt_op_no_ovl(
      dof_handler, constraints, /*overlap=*/false);
    PortableOpt::LaplaceOperatorOpt<dim, fe_degree, nq, Number> opt_op_ovl(
      dof_handler, constraints, /*overlap=*/true);

    AssertThrow(opt_op_no_ovl.m() == dof_handler.n_dofs() &&
                  opt_op_no_ovl.n() == dof_handler.n_dofs(),
                ExcMessage("Operator dimensions m()/n() mismatch!"));

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> src,
      dst_ref, dst_opt_no_ovl, dst_opt_ovl, dst_tvmult;
    ref_op.initialize_dof_vector(src);
    ref_op.initialize_dof_vector(dst_ref);
    opt_op_no_ovl.initialize_dof_vector(dst_opt_no_ovl);
    opt_op_ovl.initialize_dof_vector(dst_opt_ovl);
    opt_op_no_ovl.initialize_dof_vector(dst_tvmult);

    // Fill src deterministically across global DoF indices (including non-zero
    // values on Dirichlet DoFs to verify constraint handling)
    LinearAlgebra::ReadWriteVector<Number> rw(locally_owned);
    for (const auto idx : locally_owned)
      rw[idx] = static_cast<Number>(
        std::sin(0.37 * static_cast<double>(idx + 1)) +
        0.5 * std::cos(0.13 * static_cast<double>(idx + 3)));
    src.import_elements(rw, VectorOperation::insert);
    Kokkos::fence();

    ref_op.vmult(dst_ref, src);
    opt_op_no_ovl.vmult(dst_opt_no_ovl, src);
    opt_op_ovl.vmult(dst_opt_ovl, src);
    opt_op_no_ovl.Tvmult(dst_tvmult, src);

    const double ref_norm = dst_ref.l2_norm();
    AssertThrow(ref_norm > 0.0, ExcMessage("Reference operator norm is zero."));

    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> diff(
      dst_ref);
    diff = dst_opt_no_ovl;
    diff -= dst_ref;
    const double rel_err_no_ovl = diff.l2_norm() / ref_norm;

    diff = dst_opt_ovl;
    diff -= dst_ref;
    const double rel_err_ovl = diff.l2_norm() / ref_norm;

    diff = dst_tvmult;
    diff -= dst_ref;
    const double rel_err_tvmult = diff.l2_norm() / ref_norm;

    double rel_err_legacy = -1.0;
    if constexpr (std::is_same_v<Number, double>)
      {
        if (Utilities::MPI::n_mpi_processes(comm) == 1)
          {
            Portable::LaplaceOperator<dim, fe_degree, nq, Number> legacy_op(
              dof_handler, constraints, /*overlap=*/false);
            LinearAlgebra::distributed::Vector<Number, MemorySpace::Default>
              dst_legacy;
            legacy_op.initialize_dof_vector(dst_legacy);
            legacy_op.vmult(dst_legacy, src);
            diff = dst_legacy;
            diff -= dst_ref;
            rel_err_legacy = diff.l2_norm() / ref_norm;
          }
      }

    // Verify zero-storage GPU constant-RHS assembly (JxW = det(G) / w_3d^2)
    // against host FEValues(MappingQ(fe_degree)) on the curved, sheared mesh
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

    LinearAlgebra::ReadWriteVector<Number> rw_rhs(locally_owned);
    rw_rhs.import_elements(rhs_host, VectorOperation::insert);
    LinearAlgebra::distributed::Vector<Number, MemorySpace::Default> rhs_ref_dev,
      rhs_opt_no_ovl, rhs_opt_ovl;
    opt_op_no_ovl.initialize_dof_vector(rhs_ref_dev);
    opt_op_no_ovl.initialize_dof_vector(rhs_opt_no_ovl);
    opt_op_ovl.initialize_dof_vector(rhs_opt_ovl);
    rhs_ref_dev.import_elements(rw_rhs, VectorOperation::insert);

    opt_op_no_ovl.assemble_constant_rhs(rhs_opt_no_ovl, rhs_const);
    opt_op_ovl.assemble_constant_rhs(rhs_opt_ovl, rhs_const);

    const double rhs_ref_norm = rhs_ref_dev.l2_norm();
    diff                      = rhs_opt_no_ovl;
    diff -= rhs_ref_dev;
    const double rhs_err_no_ovl = diff.l2_norm() / rhs_ref_norm;

    diff = rhs_opt_ovl;
    diff -= rhs_ref_dev;
    const double rhs_err_ovl = diff.l2_norm() / rhs_ref_norm;

    pcout << "  [Deformed Q" << fe_degree << ", "
          << (std::is_same_v<Number, double> ? "fp64" : "fp32")
          << ", dofs=" << std::setw(5) << dof_handler.n_dofs()
          << "] rel_err(opt,no_ovl)=" << std::scientific << std::setprecision(3)
          << rel_err_no_ovl << " | rel_err(opt,ovl)=" << rel_err_ovl
          << " | rhs_err=" << std::max(rhs_err_no_ovl, rhs_err_ovl);
    if (rel_err_legacy >= 0.0)
      pcout << " | rel_err(initial_buggy)=" << rel_err_legacy;
    pcout << std::endl;

    AssertThrow(rel_err_no_ovl < rel_tol,
                ExcMessage("Deformed mesh vmult (overlap=false) failed tolerance!"));
    AssertThrow(rel_err_ovl < rel_tol,
                ExcMessage("Deformed mesh vmult (overlap=true) failed tolerance!"));
    AssertThrow(rel_err_tvmult < rel_tol,
                ExcMessage("Deformed mesh Tvmult failed tolerance!"));
    AssertThrow(rhs_err_no_ovl < rel_tol && rhs_err_ovl < rel_tol,
                ExcMessage("GPU assemble_constant_rhs failed tolerance!"));
  }

  template <int dim>
  class ManufacturedSolution : public Function<dim>
  {
  public:
    ManufacturedSolution(const Point<dim> &extents)
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
    ManufacturedRHS(const Point<dim> &extents)
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

    AffineConstraints<double> constraints;
    constraints.reinit(locally_owned, locally_relevant);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    const bool use_overlap = (Utilities::MPI::n_mpi_processes(comm) > 1);
    PortableOpt::LaplaceOperatorOpt<dim, fe_degree, nq, double> op(
      dof_handler, constraints, use_overlap);

    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> sol_dev,
      rhs_dev;
    op.initialize_dof_vector(sol_dev);
    op.initialize_dof_vector(rhs_dev);

    // Assemble manufactured RHS on host and transfer to device
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

    LinearAlgebra::ReadWriteVector<double> rw(locally_owned);
    rw.import_elements(rhs_host, VectorOperation::insert);
    rhs_dev.import_elements(rw, VectorOperation::insert);

    ReductionControl solver_control(5000, 1e-16, 1e-12);
    SolverCG<LinearAlgebra::distributed::Vector<double, MemorySpace::Default>>
      cg(solver_control);
    sol_dev = 0.0;
    cg.solve(op, sol_dev, rhs_dev, PreconditionIdentity());
    Kokkos::fence();

    // Copy solution back to host to integrate exact L2 error
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
    const Point<dim> extents(1.7, 1.1, 0.8); // Anisotropic box
    const double e1 = solve_manufactured_level<dim, fe_degree>(comm, extents, 2);
    const double e2 = solve_manufactured_level<dim, fe_degree>(comm, extents, 3);
    const double rate = std::log2(e1 / e2);

    pcout << "  [Convergence Q" << fe_degree << "] L2(ref=2)=" << std::scientific
          << std::setprecision(3) << e1 << " -> L2(ref=3)=" << e2
          << " | rate=" << std::fixed << std::setprecision(2) << rate
          << " (expected ~" << (fe_degree + 1) << ".00)" << std::endl;

    AssertThrow(rate > static_cast<double>(fe_degree) + 0.75,
                ExcMessage("Manufactured solution convergence rate too low!"));
  }
  template <int dim, int fe_degree>
  void
  run_ghost_region_indexing_test(MPI_Comm comm, ConditionalOStream &pcout)
  {
    if (Utilities::MPI::n_mpi_processes(comm) != 1)
      return;

    constexpr int          nq           = fe_degree + 2;
    constexpr unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);

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

    AffineConstraints<double> constraints;
    VectorTools::interpolate_boundary_values(MappingQ<dim>(fe_degree),
                                             dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();

    // Baseline single-domain operator evaluation
    PortableOpt::LaplaceOperatorOpt<dim, fe_degree, nq, double> opt_op(
      dof_handler, constraints, false);
    LinearAlgebra::distributed::Vector<double, MemorySpace::Default> src,
      dst_expected;
    opt_op.initialize_dof_vector(src);
    opt_op.initialize_dof_vector(dst_expected);

    const unsigned int n_dofs = dof_handler.n_dofs();
    LinearAlgebra::ReadWriteVector<double> rw(dof_handler.locally_owned_dofs());
    for (unsigned int i = 0; i < n_dofs; ++i)
      rw[i] = std::sin(0.29 * static_cast<double>(i + 1));
    src.import_elements(rw, VectorOperation::insert);
    opt_op.vmult(dst_expected, src);

    // Construct a synthetic owned + ghost buffer layout where all odd unconstrained
    // DoFs live in the ghost region [n_owned, n_owned + n_ghost) to directly
    // verify ghost-range View bounds and gather/scatter indexing (Bug 2).
    const MappingQ<dim> mapping(fe_degree);
    typename Portable::MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_gradients | update_JxW_values;
    Portable::MatrixFree<dim, double> mf;
    mf.reinit(mapping, dof_handler, constraints, QGauss<1>(nq), additional_data);
    const auto        &mf_data = mf.get_data(0);
    const unsigned int n_cells = mf_data.n_cells;

    const unsigned int n_owned = (n_dofs + 1) / 2;
    const unsigned int n_ghost = n_dofs - n_owned;

    auto map_dof = [&](const unsigned int d) -> unsigned int {
      return (d % 2 == 0) ? (d / 2) : (n_owned + (d / 2));
    };

    auto l2g_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        mf_data.local_to_global);
    BK3Opt::Parallel::DoFIndicesView remapped_dofs("remapped_dofs",
                                                   n_local_dofs,
                                                   n_cells);
    auto remapped_host = Kokkos::create_mirror_view(remapped_dofs);
    for (unsigned int c = 0; c < n_cells; ++c)
      for (unsigned int i = 0; i < n_local_dofs; ++i)
        {
          const auto gdof = l2g_host(i, c);
          remapped_host(i, c) =
            constraints.is_constrained(gdof) ?
              numbers::invalid_unsigned_int :
              map_dof(static_cast<unsigned int>(gdof));
        }
    Kokkos::deep_copy(remapped_dofs, remapped_host);

    Kokkos::View<double *, MemorySpace::Default::kokkos_space> d_in_ghosted(
      "d_in_ghosted", n_owned + n_ghost);
    Kokkos::View<double *, MemorySpace::Default::kokkos_space> d_out_ghosted(
      "d_out_ghosted", n_owned + n_ghost);
    auto h_in = Kokkos::create_mirror_view(d_in_ghosted);
    for (unsigned int d = 0; d < n_dofs; ++d)
      h_in[map_dof(d)] = rw[d];
    Kokkos::deep_copy(d_in_ghosted, h_in);
    Kokkos::deep_copy(d_out_ghosted, 0.0);

    BK3Opt::Parallel::KokkosKernel<dim, fe_degree + 1, nq, double>(
      mf_data.shape_values,
      mf_data.co_shape_gradients,
      opt_op.get_G_tensor(0),
      d_in_ghosted,
      d_out_ghosted,
      remapped_dofs,
      n_cells);

    auto h_out =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out_ghosted);
    auto h_exp = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),
      Kokkos::View<const double *, MemorySpace::Default::kokkos_space>(
        dst_expected.get_values(), n_dofs));

    double diff_sq = 0.0, norm_sq = 0.0;
    for (unsigned int d = 0; d < n_dofs; ++d)
      {
        const double got =
          constraints.is_constrained(d) ? rw[d] : h_out[map_dof(d)];
        const double err = got - h_exp[d];
        diff_sq += err * err;
        norm_sq += h_exp[d] * h_exp[d];
      }
    const double rel_err = std::sqrt(diff_sq / norm_sq);
    pcout << "  [Ghost-Region Indexing Q" << fe_degree << ", owned=" << n_owned
          << ", ghost=" << n_ghost << "] rel_err=" << std::scientific
          << std::setprecision(3) << rel_err << std::endl;
    AssertThrow(rel_err < 1e-13,
                ExcMessage("Ghost-region gather/scatter verification failed!"));

    // Also exercise the small-team fallback path (threadsPerBlock = 32 < kBatch * nq^2)
    Kokkos::deep_copy(d_out_ghosted, 0.0);
    BK3Opt::Parallel::KokkosKernel<dim, fe_degree + 1, nq, double>(
      mf_data.shape_values,
      mf_data.co_shape_gradients,
      opt_op.get_G_tensor(0),
      d_in_ghosted,
      d_out_ghosted,
      remapped_dofs,
      n_cells,
      numbers::invalid_unsigned_int,
      32u);
    auto h_out_fb =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_out_ghosted);
    double diff_fb_sq = 0.0;
    for (unsigned int d = 0; d < n_dofs; ++d)
      {
        const double got =
          constraints.is_constrained(d) ? rw[d] : h_out_fb[map_dof(d)];
        const double err = got - h_exp[d];
        diff_fb_sq += err * err;
      }
    AssertThrow(std::sqrt(diff_fb_sq / norm_sq) < 1e-13,
                ExcMessage("Small-team fallback path verification failed!"));
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

      pcout << "=== CEED BP3 Verification Suite (MPI ranks = "
            << Utilities::MPI::n_mpi_processes(comm) << ") ===" << std::endl;

      // 1. Deformed, anisotropic, sheared & curved MappingQ(p) operator tests
      run_deformed_operator_test<3, 1, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 2, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 3, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 4, double>(comm, pcout, 1e-12);
      run_deformed_operator_test<3, 6, double>(comm, pcout, 1e-12);

      // Float + odd nq (nq=3 for p=1, nq=5 for p=3) specifically exercises
      // Kokkos 8-byte ScratchMemorySpace alignment (Bug 3).
      run_deformed_operator_test<3, 1, float>(comm, pcout, 5e-5);
      run_deformed_operator_test<3, 3, float>(comm, pcout, 5e-5);

      // 2. Ghost-region [locally_owned_size, locally_owned_size + n_ghost) view
      // bounds and gather/scatter test
      run_ghost_region_indexing_test<3, 4>(comm, pcout);

      // 3. Manufactured-solution SolverCG PDE convergence on anisotropic domain
      run_convergence_test<3, 2>(comm, pcout);
      run_convergence_test<3, 3>(comm, pcout);

      pcout << "=== ALL VERIFICATION TESTS PASSED ===" << std::endl;
    }
  catch (std::exception &exc)
    {
      std::cerr << "Verification FAILED: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
