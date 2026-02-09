#ifndef bk3_kokkos_kernel_h
#define bk3_kokkos_kernel_h

#include <Kokkos_Core.hpp>

#include <vector>

namespace BK3
{
  namespace Parallel
  {
    template <typename T, const unsigned int n_q_points>
    void
    // KokkosKernel_1D_Block(
    //                     const T *__restrict__ shape_values,
    //                       const T *__restrict__ shape_gradients,
    //                       const T *__restrict__ G,
    //                       //   const T *__restrict__ in,
    //                       //   T *__restrict__ out,
    //                       const Kokkos::View<T *, Kokkos::HostSpace>
    //                       in_device, Kokkos::View<T *, Kokkos::HostSpace>
    //                       out_device, const unsigned int numThreads, const
    //                       unsigned int threadsPerBlock, const unsigned int
    //                       n_cells)
    KokkosKernel_1D_Block(
      const Kokkos::View<T *, Kokkos::HostSpace> shape_values_device,
      const Kokkos::View<T *, Kokkos::HostSpace> shape_gradients_device,
      const Kokkos::View<T *, Kokkos::HostSpace> G_device,
      const Kokkos::View<T *, Kokkos::HostSpace> in_device,
      Kokkos::View<T *, Kokkos::HostSpace>       out_device,
      const unsigned int                         numThreads,
      const unsigned int                         threadsPerBlock,
      const unsigned int                         n_cells)
    {
      unsigned int numBlocks =
        numThreads /
        (std::min(n_q_points * n_q_points * n_q_points, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;

      const unsigned int n_dofs = n_q_points;

      {
        // Kokkos::View<const T *, Kokkos::HostSpace> shape_values_view(
        //   shape_values, n_dofs * n_q_points);
        // Kokkos::View<T *> shape_values_device("shape_values_device",
        //                                       n_dofs * n_q_points);
        // Kokkos::deep_copy(shape_values_device, shape_values_view);

        // Kokkos::View<const T *, Kokkos::HostSpace> shape_gradients_view(
        //   shape_gradients, n_q_points * n_q_points);
        // Kokkos::View<T *> shape_gradients_device("shape_gradients_device",
        //                                          n_q_points * n_q_points);
        // Kokkos::deep_copy(shape_gradients_device, shape_gradients_view);

        // we probably don't want to copy it each iteration!
        // Kokkos::View<const T *, Kokkos::HostSpace> G_view(
        //   G, n_cells * n_q_points * n_q_points * n_q_points * 6);
        // Kokkos::View<T *> G_device("G_device",
        //                       n_cells * n_q_points * n_q_points * n_q_points
        //                       *
        //                         6);
        // Kokkos::deep_copy(G_device, G_view);

        // Kokkos::View<const T *, Kokkos::HostSpace> in_view(in,
        //                                                    n_cells * n_dofs *
        //                                                      n_dofs *
        //                                                      n_dofs);
        // Kokkos::View<T *>                          in_device("in_device",
        //                             n_cells * n_dofs * n_dofs * n_dofs);
        // Kokkos::deep_copy(in_device, in_view);

        // Kokkos::View<const T *, Kokkos::HostSpace> out_view(out,
        //                                                     n_cells * n_dofs
        //                                                     *
        //                                                       n_dofs *
        //                                                       n_dofs);
        // Kokkos::View<T *>                          out_device("out_device",
        //                              n_cells * n_dofs * n_dofs * n_dofs);


        // Kokkos with shared memory
        unsigned int ssize = 3 * n_dofs * n_q_points +
                             3 * n_q_points * n_q_points +
                             5 * n_q_points * n_q_points * n_q_points;

        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        // kokkosTimer.start();
        Kokkos::parallel_for(
          policy, KOKKOS_LAMBDA(member_type team_member) {

            // shared memory access
            T *scratch = (T *)team_member.team_shmem().get_shmem(shmem_size);
            T *shape_values_scratch = scratch;
            T *shape_gradients_scratch =
              shape_values_scratch + n_dofs * n_q_points;
            T *rqr    = shape_gradients_scratch + n_q_points * n_q_points;
            T *rqs    = rqr + n_q_points * n_q_points * n_q_points;
            T *rqt    = rqs + n_q_points * n_q_points * n_q_points;
            T *s_wsp0 = rqt + n_q_points * n_q_points * n_q_points;
            T *s_wsp1 = s_wsp0 + n_q_points * n_q_points * n_q_points;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();

            // copy to shared memory
            for (unsigned int tid = threadIdx; tid < n_dofs * n_q_points;
                 tid += blockSize)
              {
                shape_values_scratch[tid] = shape_values_device[tid];
              }

            for (unsigned int tid = threadIdx; tid < n_q_points * n_q_points;
                 tid += blockSize)
              {
                shape_gradients_scratch[tid] = shape_gradients_device[tid];
              }

            team_member.team_barrier();

            /*
            Interpolate to GL nodes
            */

            // element index
            unsigned int e = team_member.league_rank();


            while (e < n_cells)
              {
                // step-1 : Copy from in to the wsp0
                for (unsigned int tid = threadIdx;
                     tid < n_dofs * n_dofs * n_dofs;
                     tid += blockSize)
                  {
                    s_wsp0[tid] = in_device[e * n_dofs * n_dofs * n_dofs + tid];
                  }
                team_member.team_barrier();

                // step-2 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_dofs * n_dofs;
                     tid += blockSize)
                  {
                    const int p = tid / (n_dofs * n_dofs);
                    const int j = (tid % (n_dofs * n_dofs)) / n_dofs;
                    const int k = tid % n_dofs;

                    T tmp = 0.0;
                    for (unsigned int i = 0; i < n_dofs; ++i)
                      {
                        tmp += s_wsp0[i * n_dofs * n_dofs + j * n_dofs + k] *
                               shape_values_scratch[p * n_dofs + i];
                      }
                    s_wsp1[p * n_dofs * n_dofs + j * n_dofs + k] = tmp;
                  }
                team_member.team_barrier();

                // step-3 : direction 1
                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_q_points * n_dofs;
                     tid += blockSize)
                  {
                    const int q = tid / (n_q_points * n_dofs);
                    const int p = (tid % (n_q_points * n_dofs)) / n_dofs;
                    const int k = tid % n_dofs;

                    T tmp = 0.0;
                    for (unsigned int j = 0; j < n_dofs; j++)
                      {
                        tmp += s_wsp1[p * n_dofs * n_dofs + j * n_dofs + k] *
                               shape_values_scratch[q * n_dofs + j];
                      }
                    s_wsp0[q * n_q_points * n_dofs + p * n_dofs + k] = tmp;
                  }
                team_member.team_barrier();

                // step-4 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_q_points * n_q_points;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points * n_q_points);
                    const int q =
                      (tid % (n_q_points * n_q_points)) / n_q_points;
                    const int r = tid % n_q_points;

                    T tmp = 0.0;
                    for (unsigned int k = 0; k < n_dofs; ++k)
                      {
                        tmp +=
                          s_wsp0[q * n_q_points * n_dofs + p * n_dofs + k] *
                          shape_values_scratch[r * n_dofs + k];
                      }
                    s_wsp1[p * n_q_points * n_q_points + q * n_q_points + r] =
                      tmp;
                  }
                team_member.team_barrier();

                // Geometric vals
                T Grr, Grs, Grt, Gss, Gst, Gtt;
                T qr, qs, qt;

                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_q_points * n_q_points;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points * n_q_points);
                    const int q =
                      (tid % (n_q_points * n_q_points)) / n_q_points;
                    const int r = tid % n_q_points;

                    qr = 0;
                    qs = 0;
                    qt = 0;

                    // step-5 : Load Geometric Factors, coalesced access
                    Grr = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              0 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];
                    Grs = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              1 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];
                    Grt = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              2 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];
                    Gss = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              3 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];
                    Gst = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              4 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];
                    Gtt = G_device[e * 6 * n_q_points * n_q_points * n_q_points +
                              5 * n_q_points * n_q_points * n_q_points +
                              p * n_q_points * n_q_points + q * n_q_points + r];

                    // step-6 : Multiply by D
                    for (unsigned int n = 0; n < n_q_points; n++)
                      {
                        qr += s_wsp1[n * n_q_points * n_q_points +
                                     q * n_q_points + r] *
                              shape_gradients_scratch[p * n_q_points + n];
                      }

                    for (unsigned int n = 0; n < n_q_points; n++)
                      {
                        qs += s_wsp1[p * n_q_points * n_q_points +
                                     n * n_q_points + r] *
                              shape_gradients_scratch[q * n_q_points + n];
                      }

                    for (unsigned int n = 0; n < n_q_points; n++)
                      {
                        qt += s_wsp1[p * n_q_points * n_q_points +
                                     q * n_q_points + n] *
                              shape_gradients_scratch[r * n_q_points + n];
                      }

                    // step-7 : Apply chain rule
                    rqr[p * n_q_points * n_q_points + q * n_q_points + r] =
                      Grr * qr + Grs * qs + Grt * qt;
                    rqs[p * n_q_points * n_q_points + q * n_q_points + r] =
                      Grs * qr + Gss * qs + Gst * qt;
                    rqt[p * n_q_points * n_q_points + q * n_q_points + r] =
                      Grt * qr + Gst * qs + Gtt * qt;
                  }
                team_member.team_barrier();


                // step-8 : Compute out vector in GL nodes
                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_q_points * n_q_points;
                     tid += blockSize)
                  {
                    const int p = tid / (n_q_points * n_q_points);
                    const int q =
                      (tid % (n_q_points * n_q_points)) / n_q_points;
                    const int r = tid % n_q_points;

                    T tmp0 = 0;
                    for (unsigned int n = 0; n < n_q_points; ++n)
                      tmp0 +=
                        rqr[n * n_q_points * n_q_points + q * n_q_points + r] *
                        shape_gradients_scratch[n * n_q_points + p];

                    for (unsigned int n = 0; n < n_q_points; ++n)
                      tmp0 +=
                        rqs[p * n_q_points * n_q_points + n * n_q_points + r] *
                        shape_gradients_scratch[n * n_q_points + q];

                    for (unsigned int n = 0; n < n_q_points; ++n)
                      tmp0 +=
                        rqt[p * n_q_points * n_q_points + q * n_q_points + n] *
                        shape_gradients_scratch[n * n_q_points + r];

                    s_wsp1[p * n_q_points * n_q_points + q * n_q_points + r] =
                      tmp0;
                  }
                team_member.team_barrier();


                /*
                Interpolate to GLL nodes
                */

                // step-9 : direction 2
                for (unsigned int tid = threadIdx;
                     tid < n_q_points * n_q_points * n_dofs;
                     tid += blockSize)
                  {
                    const int q = tid / (n_q_points * n_dofs);
                    const int p = (tid % (n_q_points * n_dofs)) / n_dofs;
                    const int k = tid % n_dofs;

                    T tmp = 0.0;
                    for (unsigned int r = 0; r < n_q_points; ++r)
                      {
                        tmp += s_wsp1[p * n_q_points * n_q_points +
                                      q * n_q_points + r] *
                               shape_values_scratch[r * n_dofs + k];
                      }
                    s_wsp0[q * n_q_points * n_dofs + p * n_dofs + k] = tmp;
                  }
                team_member.team_barrier();

                // step-10 : dirco_gradientsection 1
                for (unsigned int tid = threadIdx;
                     tid < n_dofs * n_dofs * n_q_points;
                     tid += blockSize)
                  {
                    const int p = tid / (n_dofs * n_dofs);
                    const int j = (tid % (n_dofs * n_dofs)) / n_dofs;
                    const int k = tid % n_dofs;

                    T tmp = 0.0;
                    for (unsigned int q = 0; q < n_q_points; q++)
                      {
                        tmp +=
                          s_wsp0[q * n_q_points * n_dofs + p * n_dofs + k] *
                          shape_values_scratch[q * n_dofs + j];
                      }
                    s_wsp1[p * n_dofs * n_dofs + j * n_dofs + k] = tmp;
                  }
                team_member.team_barrier();

                // step-11 : direction 0
                for (unsigned int tid = threadIdx;
                     tid < n_dofs * n_dofs * n_dofs;
                     tid += blockSize)
                  {
                    const int i = tid / (n_dofs * n_dofs);
                    const int j = (tid % (n_dofs * n_dofs)) / n_dofs;
                    const int k = tid % n_dofs;

                    T tmp = 0.0;
                    for (unsigned int p = 0; p < n_q_points; ++p)
                      {
                        tmp += s_wsp1[p * n_dofs * n_dofs + j * n_dofs + k] *
                               shape_values_scratch[p * n_dofs + i];
                      }
                    s_wsp0[i * n_dofs * n_dofs + j * n_dofs + k] = tmp;
                  }
                team_member.team_barrier();

                // step-12 : Copy wsp0 to out
                for (unsigned int tid = threadIdx;
                     tid < n_dofs * n_dofs * n_dofs;
                     tid += blockSize)
                  {
                    out_device[e * n_dofs * n_dofs * n_dofs + tid] =
                      s_wsp0[tid];
                  }
                team_member.team_barrier();

                e += team_member.league_size();
              }
          });
        Kokkos::fence();
      }
    }



  } // namespace Parallel
} // namespace BK3

#endif