#ifndef bk3_kokkos_kernel_opt_h
#define bk3_kokkos_kernel_opt_h

#include <deal.II/base/memory_space.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/utilities.h>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>

DEAL_II_NAMESPACE_OPEN

namespace BK3Opt
{
  namespace Parallel
  {
    using ExecutionSpace =
      dealii::MemorySpace::Default::kokkos_space::execution_space;

    template <typename number>
    using DeviceView =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>;

    template <typename number>
    using ConstDeviceView =
      Kokkos::View<const number *, MemorySpace::Default::kokkos_space>;

    template <typename number>
    using RandomAccessDeviceView =
      Kokkos::View<const number *,
                   MemorySpace::Default::kokkos_space,
                   Kokkos::MemoryTraits<Kokkos::RandomAccess>>;

    template <typename number>
    using SharedView = Kokkos::View<number *,
                                    ExecutionSpace::scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    using DoFIndicesView =
      Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>;

    using ConstDoFIndicesView =
      Kokkos::View<const unsigned int **, MemorySpace::Default::kokkos_space>;

    template <int dim, int nm, int nq, typename number>
    struct KokkosKernelFunctor
    {
      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      using MemberType = typename TeamPolicy::member_type;

      static constexpr unsigned int nq_total = Utilities::pow(nq, dim);
      static constexpr unsigned int nm_total = Utilities::pow(nm, dim);

      ConstDeviceView<number>        d_shape_values;
      ConstDeviceView<number>        d_co_shape_gradients;
      ConstDeviceView<number>        d_G;
      RandomAccessDeviceView<number> d_in;
      DeviceView<number>             d_out;
      ConstDoFIndicesView            dof_indices;
      unsigned int                   n_cells;
      unsigned int                   nelmtPerBatch;
      unsigned int                   total_batches;
      std::size_t                    shmem_size;
      bool                           use_coloring;

      std::size_t
      team_shmem_size(int /*team_size*/) const
      {
        return shmem_size;
      }

      KOKKOS_INLINE_FUNCTION void
      operator()(const MemberType &team_member) const
      {
        number r_p[nq];
        number r_q[nq];
        number r_r[nq];

        SharedView<number> s_shape_values(team_member.team_shmem(), nm * nq);
        SharedView<number> s_co_shape_gradients(team_member.team_shmem(),
                                                nq * nq);

        SharedView<number> s_wsp0(team_member.team_shmem(),
                                  nelmtPerBatch * nq_total);
        SharedView<number> s_wsp1(team_member.team_shmem(),
                                  nelmtPerBatch * nq_total);

        SharedView<number> s_rqr(team_member.team_shmem(),
                                 nelmtPerBatch * nq_total);
        SharedView<number> s_rqs(team_member.team_shmem(),
                                 nelmtPerBatch * nq_total);
        SharedView<number> s_rqt = s_wsp0;

        const unsigned int threadIdx = team_member.team_rank();
        const unsigned int blockSize = team_member.team_size();

        // copy to shared memory
        for (unsigned int tid = threadIdx;
             tid < static_cast<unsigned int>(nm * nq);
             tid += blockSize)
          {
            s_shape_values[tid] = d_shape_values[tid];
          }

        for (unsigned int tid = threadIdx;
             tid < static_cast<unsigned int>(nq * nq);
             tid += blockSize)
          {
            s_co_shape_gradients[tid] = d_co_shape_gradients[tid];
          }
        team_member.team_barrier();

        /*
        Interpolate to GL nodes
        */

        // element batch iteration
        unsigned int eb = team_member.league_rank();
        while (eb < total_batches)
          {
            const unsigned int c_nelmtPerBatch =
              (eb * nelmtPerBatch + nelmtPerBatch > n_cells) ?
                (n_cells - eb * nelmtPerBatch) :
                nelmtPerBatch;

            // step-1 : Copy from in to the scratch values
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nm);
                 tid += blockSize)
              {
                const int e = tid / (nm * nm);
                const int j = (tid % (nm * nm)) / nm;
                const int k = tid % nm;

                const unsigned int global_cell_index = eb * nelmtPerBatch + e;

                for (int i = 0; i < nm; ++i)
                  {
                    const int local_idx = i * nm * nm + j * nm + k;
                    const unsigned int dof_index =
                      dof_indices(local_idx, global_cell_index);
                    const int shared_idx = e * nm_total + local_idx;

                    if (dof_index == numbers::invalid_unsigned_int)
                      s_wsp0[shared_idx] = 0;
                    else
                      s_wsp0[shared_idx] = d_in[dof_index];
                  }
              }
            team_member.team_barrier();

            // step-2 : direction 0 (contract i -> p)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nm);
                 tid += blockSize)
              {
                int e = tid / (nm * nm);
                int j = (tid % (nm * nm)) / nm;
                int k = tid % nm;

                for (int i = 0; i < nm; ++i)
                  {
                    r_p[i] = s_wsp0[e * nm * nm * nm + i * nm * nm + j * nm + k];
                  }

                for (int p = 0; p < nq; ++p)
                  {
                    number tmp = 0.0;
                    for (int i = 0; i < nm; ++i)
                      {
                        tmp += s_shape_values[i * nq + p] * r_p[i];
                      }
                    s_wsp1[e * nq * nm * nm + p * nm * nm + j * nm + k] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-3 : direction 1 (contract j -> q)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nq);
                 tid += blockSize)
              {
                int e = tid / (nq * nm);
                int p = (tid % (nq * nm)) / nm;
                int k = tid % nm;
                for (int j = 0; j < nm; ++j)
                  {
                    r_q[j] = s_wsp1[e * nq * nm * nm + p * nm * nm + j * nm + k];
                  }

                for (int q = 0; q < nq; ++q)
                  {
                    number tmp = 0.0;
                    for (int j = 0; j < nm; ++j)
                      {
                        tmp += s_shape_values[j * nq + q] * r_q[j];
                      }
                    s_wsp0[e * nq * nq * nm + q * nq * nm + p * nm + k] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-4 : direction 2 (contract k -> r)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nq * nq);
                 tid += blockSize)
              {
                int e = tid / (nq * nq);
                int q = (tid % (nq * nq)) / nq;
                int p = tid % nq;

                for (int k = 0; k < nm; ++k)
                  {
                    r_r[k] = s_wsp0[e * nq * nq * nm + q * nq * nm + p * nm + k];
                  }
                for (int r = 0; r < nq; ++r)
                  {
                    number tmp = 0.0;
                    for (int k = 0; k < nm; ++k)
                      {
                        tmp += s_shape_values[k * nq + r] * r_r[k];
                      }
                    s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-5 : gradients & metric contraction
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nq * nq);
                 tid += blockSize)
              {
                int e = tid / (nq * nq);
                int q = (tid % (nq * nq)) / nq;
                int r = tid % nq;

                // copy to register
                for (unsigned int n = 0; n < static_cast<unsigned int>(nq); n++)
                  {
                    r_p[n] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                    r_q[n] = s_co_shape_gradients[n * nq + q];
                    r_r[n] = s_co_shape_gradients[n * nq + r];
                  }

                number Grr, Grs, Grt, Gss, Gst, Gtt;
                number qr, qs, qt;

                const unsigned int cell_id = eb * nelmtPerBatch + e;

                for (unsigned int p = 0; p < static_cast<unsigned int>(nq); ++p)
                  {
                    qr = 0;
                    qs = 0;
                    qt = 0;

                    const std::size_t q_pt =
                      static_cast<std::size_t>(p * nq * nq + q * nq + r);
                    const std::size_t g_base =
                      static_cast<std::size_t>(cell_id) * 6 * nq_total;

                    Grr = d_G[g_base + 0 * nq_total + q_pt];
                    Grs = d_G[g_base + 1 * nq_total + q_pt];
                    Grt = d_G[g_base + 2 * nq_total + q_pt];
                    Gss = d_G[g_base + 3 * nq_total + q_pt];
                    Gst = d_G[g_base + 4 * nq_total + q_pt];
                    Gtt = d_G[g_base + 5 * nq_total + q_pt];

                    // Multiply by D
                    for (unsigned int n = 0; n < static_cast<unsigned int>(nq); n++)
                      {
                        qr += s_co_shape_gradients[n * nq + p] * r_p[n];
                        qs += r_q[n] * s_wsp1[e * nq * nq * nq + r * nq * nq + n * nq + p];
                        qt += r_r[n] * s_wsp1[e * nq * nq * nq + n * nq * nq + q * nq + p];
                      }

                    // Apply chain rule (CORRECTED AXIS MAPPING):
                    // In deal.II: axis 0 is 'r' (from k), axis 1 is 'q' (from j), axis 2 is 'p' (from i).
                    // qt is derivative along axis 0, qs is along axis 1, qr is along axis 2.
                    const number g0 = qt; // axis 0 (r)
                    const number g1 = qs; // axis 1 (q)
                    const number g2 = qr; // axis 2 (p)

                    s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                      Grr * g0 + Grs * g1 + Grt * g2; // flux 0 (along r)
                    s_rqs[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                      Grs * g0 + Gss * g1 + Gst * g2; // flux 1 (along q)
                    s_rqr[e * nq * nq * nq + p * nq * nq + q * nq + r] =
                      Grt * g0 + Gst * g1 + Gtt * g2; // flux 2 (along p)
                  }
              }
            team_member.team_barrier();

            // step-6 : apply transpose collocation derivatives D^T
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nq * nq);
                 tid += blockSize)
              {
                int e = tid / (nq * nq);
                int q = (tid % (nq * nq)) / nq;
                int r = tid % nq;

                // copy to register
                for (unsigned int n = 0; n < static_cast<unsigned int>(nq); n++)
                  {
                    r_p[n] = s_rqr[e * nq * nq * nq + n * nq * nq + q * nq + r];
                    r_q[n] = s_co_shape_gradients[q * nq + n];
                    r_r[n] = s_co_shape_gradients[r * nq + n];
                  }

                for (unsigned int p = 0; p < static_cast<unsigned int>(nq); ++p)
                  {
                    number tmp0 = 0;
                    for (unsigned int n = 0; n < static_cast<unsigned int>(nq); ++n)
                      tmp0 += r_p[n] * s_co_shape_gradients[p * nq + n];

                    for (unsigned int n = 0; n < static_cast<unsigned int>(nq); ++n)
                      tmp0 += s_rqs[e * nq * nq * nq + p * nq * nq + n * nq + r] * r_q[n];

                    for (unsigned int n = 0; n < static_cast<unsigned int>(nq); ++n)
                      tmp0 += s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + n] * r_r[n];

                    s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p] = tmp0;
                  }
              }
            team_member.team_barrier();

            /*
            Interpolate to GLL nodes
            */

            // step-9 : direction 2 (contract r -> k)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nq * nq);
                 tid += blockSize)
              {
                int e = tid / (nq * nq);
                int q = (tid % (nq * nq)) / nq;
                int p = tid % nq;

                for (int r = 0; r < nq; ++r)
                  {
                    r_r[r] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + p];
                  }

                for (int k = 0; k < nm; ++k)
                  {
                    number tmp = 0.0;
                    for (int r = 0; r < nq; ++r)
                      {
                        tmp += s_shape_values[k * nq + r] * r_r[r];
                      }
                    s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-10 : direction 1 (contract q -> j)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nq);
                 tid += blockSize)
              {
                int e = tid / (nm * nq);
                int k = (tid % (nm * nq)) / nq;
                int p = tid % nq;

                for (int q = 0; q < nq; ++q)
                  {
                    r_q[q] = s_wsp0[e * nm * nq * nq + k * nq * nq + q * nq + p];
                  }

                for (int j = 0; j < nm; ++j)
                  {
                    number tmp = 0.0;
                    for (int q = 0; q < nq; ++q)
                      {
                        tmp += s_shape_values[j * nq + q] * r_q[q];
                      }
                    s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-11 : direction 0 (contract p -> i)
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nm);
                 tid += blockSize)
              {
                int e = tid / (nm * nm);
                int j = (tid % (nm * nm)) / nm;
                int k = tid % nm;

                for (int p = 0; p < nq; ++p)
                  {
                    r_p[p] = s_wsp1[e * nm * nm * nq + k * nm * nq + j * nq + p];
                  }

                for (int i = 0; i < nm; ++i)
                  {
                    number tmp = 0.0;
                    for (int p = 0; p < nq; ++p)
                      {
                        tmp += s_shape_values[i * nq + p] * r_p[p];
                      }
                    s_wsp0[e * nm * nm * nm + i * nm * nm + j * nm + k] = tmp;
                  }
              }
            team_member.team_barrier();

            // step-12 : Copy wsp0 (result) back to global out vector
            for (unsigned int tid = threadIdx;
                 tid < c_nelmtPerBatch * static_cast<unsigned int>(nm * nm);
                 tid += blockSize)
              {
                const int e = tid / (nm * nm);
                const int j = (tid % (nm * nm)) / nm;
                const int k = tid % nm;

                const unsigned int global_cell_index = eb * nelmtPerBatch + e;

                for (int i = 0; i < nm; ++i)
                  {
                    const int local_idx = i * nm * nm + j * nm + k;
                    const unsigned int dof_index =
                      dof_indices(local_idx, global_cell_index);
                    const int shared_idx = e * nm_total + local_idx;

                    if (dof_index != numbers::invalid_unsigned_int)
                      {
                        if (use_coloring)
                          d_out[dof_index] += s_wsp0[shared_idx];
                        else
                          Kokkos::atomic_add(&d_out[dof_index], s_wsp0[shared_idx]);
                      }
                  }
              }
            team_member.team_barrier();

            eb += team_member.league_size();
          }
      }
    };

    template <int dim, int nm, int nq, typename number>
    void
    KokkosKernelAsync(
      const ExecutionSpace                &exec,
      const ConstDeviceView<number>        d_shape_values,
      const ConstDeviceView<number>        d_co_shape_gradients,
      const ConstDeviceView<number>        d_G,
      const RandomAccessDeviceView<number> d_in,
      DeviceView<number>                   d_out,
      const ConstDoFIndicesView            dof_indices,
      const unsigned int                   n_cells,
      unsigned int numBlocks       = numbers::invalid_unsigned_int,
      unsigned int threadsPerBlock = numbers::invalid_unsigned_int,
      const bool   use_coloring    = false)
    {
      static_assert(dim == 3, "BK3Opt::Parallel::KokkosKernel requires dim == 3.");
      if (n_cells == 0)
        return;

      constexpr unsigned int nq_total = Utilities::pow(nq, dim);
      constexpr bool is_serial =
        std::is_same<ExecutionSpace, Kokkos::DefaultHostExecutionSpace>::value;

      // finding the batch size (initial 10.8 KB shmemPerBlock)
      int shmemPerBlock = 10800;
      unsigned int nelmtPerBatch =
        shmemPerBlock / (4 * nq_total) / sizeof(number);
      if (nelmtPerBatch == 0)
        nelmtPerBatch = 1;

      if (is_serial)
        nelmtPerBatch = 1;

      const unsigned int total_batches =
        (n_cells + nelmtPerBatch - 1u) / nelmtPerBatch;

      if (numBlocks == numbers::invalid_unsigned_int)
        numBlocks = is_serial ? 1u : std::max(1u, (n_cells + nelmtPerBatch - 1u) / nelmtPerBatch / 2u);

      if (numBlocks == 0)
        numBlocks = 1u;

      if (threadsPerBlock == numbers::invalid_unsigned_int)
        threadsPerBlock = is_serial ? 1u : (nq * nq * std::max(1u, nelmtPerBatch));

      const std::size_t shmem_size =
        SharedView<number>::shmem_size(nm * nq) +
        SharedView<number>::shmem_size(nq * nq) +
        4u * SharedView<number>::shmem_size(nelmtPerBatch * nq_total);

      using FunctorType = KokkosKernelFunctor<dim, nm, nq, number>;
      FunctorType functor{d_shape_values,
                          d_co_shape_gradients,
                          d_G,
                          d_in,
                          d_out,
                          dof_indices,
                          n_cells,
                          nelmtPerBatch,
                          total_batches,
                          shmem_size,
                          use_coloring};

      using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
      TeamPolicy probe_policy(exec, 1, Kokkos::AUTO);
      const unsigned int max_team = static_cast<unsigned int>(
        probe_policy.team_size_max(functor, Kokkos::ParallelForTag()));
      threadsPerBlock = std::max(1u, std::min(threadsPerBlock, max_team));

      TeamPolicy policy(exec, numBlocks, threadsPerBlock);
      Kokkos::parallel_for("BK3Opt::KokkosKernel", policy, functor);
    }

    template <int dim, int nm, int nq, typename number>
    void
    KokkosKernel(const ConstDeviceView<number>        d_shape_values,
                 const ConstDeviceView<number>        d_co_shape_gradients,
                 const ConstDeviceView<number>        d_G,
                 const RandomAccessDeviceView<number> d_in,
                 DeviceView<number>                   d_out,
                 const ConstDoFIndicesView            dof_indices,
                 const unsigned int                   n_cells,
                 unsigned int numBlocks       = numbers::invalid_unsigned_int,
                 unsigned int threadsPerBlock = numbers::invalid_unsigned_int,
                 const bool   use_coloring    = false)
    {
      ExecutionSpace exec;
      KokkosKernelAsync<dim, nm, nq, number>(exec,
                                             d_shape_values,
                                             d_co_shape_gradients,
                                             d_G,
                                             d_in,
                                             d_out,
                                             dof_indices,
                                             n_cells,
                                             numBlocks,
                                             threadsPerBlock,
                                             use_coloring);
      exec.fence();
    }

  } // namespace Parallel
} // namespace BK3Opt

DEAL_II_NAMESPACE_CLOSE

#endif
