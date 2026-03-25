#ifndef portable_laplace_operator_base_h
#define portable_laplace_operator_base_h

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/memory_space.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/portable_matrix_free.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable
{
  template <int dim, typename number>
  class LaplaceOperatorBase : public EnableObserverPointer
  {
  public:
    virtual ~LaplaceOperatorBase() = default;

    virtual void
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const = 0;

    virtual void
    Tvmult(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
        &src) const = 0;

    virtual void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
      const = 0;

    virtual types::global_dof_index
    m() const = 0;

    virtual types::global_dof_index
    n() const = 0;

    virtual const Portable::MatrixFree<dim, number> &
    get_matrix_free() const = 0;

    virtual const std::shared_ptr<const Utilities::MPI::Partitioner> &
    get_vector_partitioner() const = 0;
  };

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
