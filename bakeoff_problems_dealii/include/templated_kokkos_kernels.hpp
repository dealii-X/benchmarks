#ifndef BK3_TEMPLATED_KOKKOS_KERNELS_HPP
#define BK3_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>

#include <vector>

namespace BK3
{
  namespace Parallel
  {
    template <typename T,
              const unsigned int nq0,
              const unsigned int nq1,
              const unsigned int nq2>
    void
    KokkosKernel_1D_Block(const T *__restrict__ basis0,
                          const T *__restrict__ basis1,
                          const T *__restrict__ basis2,
                          const T *__restrict__ dbasis0,
                          const T *__restrict__ dbasis1,
                          const T *__restrict__ dbasis2,
                          const T *__restrict__ G,
                          const T *__restrict__ in,
                          T *__restrict__ out,
                          const unsigned int numThreads,
                          const unsigned int threadsPerBlock,
                          const unsigned int nelmt,
                          const unsigned int ntests)
    {
      unsigned int numBlocks =
        numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
      if (numBlocks == 0)
        numBlocks = 1;

      const unsigned int nm0 = nq0;
      const unsigned int nm1 = nq1;
      const unsigned int nm2 = nq2;

      std::vector<T> results(3);
      {
        Kokkos::View<const T *, Kokkos::HostSpace> basis0_view(basis0,
                                                               nm0 * nq0);
        Kokkos::View<const T *, Kokkos::HostSpace> basis1_view(basis1,
                                                               nm1 * nq1);
        Kokkos::View<const T *, Kokkos::HostSpace> basis2_view(basis2,
                                                               nm2 * nq2);
        Kokkos::View<T *> d_basis0("d_basis0", nm0 * nq0);
        Kokkos::View<T *> d_basis1("d_basis1", nm1 * nq1);
        Kokkos::View<T *> d_basis2("d_basis2", nm2 * nq2);
        Kokkos::deep_copy(d_basis0, basis0_view);
        Kokkos::deep_copy(d_basis1, basis1_view);
        Kokkos::deep_copy(d_basis2, basis2_view);

        Kokkos::View<const T *, Kokkos::HostSpace> dbasis0_view(dbasis0,
                                                                nq0 * nq0);
        Kokkos::View<const T *, Kokkos::HostSpace> dbasis1_view(dbasis1,
                                                                nq1 * nq1);
        Kokkos::View<const T *, Kokkos::HostSpace> dbasis2_view(dbasis2,
                                                                nq2 * nq2);
        Kokkos::View<T *> d_dbasis0("d_dbasis0", nq0 * nq0);
        Kokkos::View<T *> d_dbasis1("d_dbasis1", nq1 * nq1);
        Kokkos::View<T *> d_dbasis2("d_dbasis2", nq2 * nq2);
        Kokkos::deep_copy(d_dbasis0, dbasis0_view);
        Kokkos::deep_copy(d_dbasis1, dbasis1_view);
        Kokkos::deep_copy(d_dbasis2, dbasis2_view);

        Kokkos::View<const T *, Kokkos::HostSpace> G_view(G,
                                                          nelmt * nq0 * nq1 *
                                                            nq2 * 6);
        Kokkos::View<T *> d_G("d_G", nelmt * nq0 * nq1 * nq2 * 6);
        Kokkos::deep_copy(d_G, G_view);

        Kokkos::View<const T *, Kokkos::HostSpace> in_view(in,
                                                           nelmt * nm0 * nm1 *
                                                             nm2);
        Kokkos::View<T *> d_in("d_in", nelmt * nm0 * nm1 * nm2);
        Kokkos::deep_copy(d_in, in_view);

        Kokkos::View<const T *, Kokkos::HostSpace> out_view(out,
                                                            nelmt * nm0 * nm1 *
                                                              nm2);
        Kokkos::View<T *> d_out("d_out", nelmt * nm0 * nm1 * nm2);


        // Kokkos with shared memory
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq0 +
                             nq1 * nq1 + nq2 * nq2 + 5 * nq0 * nq1 * nq2;

        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        for (unsigned int t = 0u; t < ntests; ++t)
          {
            Kokkos::parallel_for(
              policy, KOKKOS_LAMBDA(member_type team_member) {
                // T r_p[10];
                // T r_q[10];
                // T r_r[10];

                // shared memory access
                T *scratch =
                  (T *)team_member.team_shmem().get_shmem(shmem_size);
                T *s_basis0  = scratch;
                T *s_basis1  = s_basis0 + nm0 * nq0;
                T *s_basis2  = s_basis1 + nm1 * nq1;
                T *s_dbasis0 = s_basis2 + nm2 * nq2;
                T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
                T *s_dbasis2 = s_dbasis1 + nq1 * nq1;
                T *rqr       = s_dbasis2 + nq2 * nq2;
                T *rqs       = rqr + nq0 * nq1 * nq2;
                T *rqt       = rqs + nq0 * nq1 * nq2;
                T *s_wsp0    = rqt + nq0 * nq1 * nq2;
                T *s_wsp1    = s_wsp0 + nq0 * nq1 * nq2;

                const unsigned int threadIdx = team_member.team_rank();
                const unsigned int blockSize = team_member.team_size();

                // copy to shared memory
                for (unsigned int tid = threadIdx; tid < nm0 * nq0;
                     tid += blockSize)
                  {
                    s_basis0[tid] = d_basis0[tid];
                  }

                for (unsigned int tid = threadIdx; tid < nm1 * nq1;
                     tid += blockSize)
                  {
                    s_basis1[tid] = d_basis1[tid];
                  }

                for (unsigned int tid = threadIdx; tid < nm2 * nq2;
                     tid += blockSize)
                  {
                    s_basis2[tid] = d_basis2[tid];
                  }


                for (unsigned int tid = threadIdx; tid < nq0 * nq0;
                     tid += blockSize)
                  {
                    s_dbasis0[tid] = d_dbasis0[tid];
                  }


                for (unsigned int tid = threadIdx; tid < nq1 * nq1;
                     tid += blockSize)
                  {
                    s_dbasis1[tid] = d_dbasis1[tid];
                  }

                for (unsigned int tid = threadIdx; tid < nq2 * nq2;
                     tid += blockSize)
                  {
                    s_dbasis2[tid] = d_dbasis2[tid];
                  }

                team_member.team_barrier();

                /*
                Interpolate to GL nodes
                */

                // element index
                unsigned int e = team_member.league_rank();


                while (e < nelmt)
                  {
                    // step-1 : Copy from in to the wsp0
                    for (unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2;
                         tid += blockSize)
                      {
                        s_wsp0[tid] = d_in[e * nm0 * nm1 * nm2 + tid];
                        // Kokkos::printf("%f, ", s_wsp0[tid]);
                      }
                    team_member.team_barrier();
                    // std::cout << std::endl;

                    // step-2 : direction 0
                    for (unsigned int tid = threadIdx; tid < nq0 * nm1 * nm2;
                         tid += blockSize)
                      {
                        const int p = tid / (nm1 * nm2);
                        const int j = (tid % (nm1 * nm2)) / nm2;
                        const int k = tid % nm2;

                        T tmp = 0.0;
                        for (unsigned int i = 0; i < nm0; ++i)
                          {
                            tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] *
                                   s_basis0[p * nm0 + i];
                          }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                      }
                    team_member.team_barrier();

                    // step-3 : direction 1
                    for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2;
                         tid += blockSize)
                      {
                        const int q = tid / (nq0 * nm2);
                        const int p = (tid % (nq0 * nm2)) / nm2;
                        const int k = tid % nm2;

                        T tmp = 0.0;
                        for (unsigned int j = 0; j < nm1; j++)
                          {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] *
                                   s_basis1[q * nm1 + j];
                          }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                      }
                    team_member.team_barrier();

                    // step-4 : direction 2
                    for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2;
                         tid += blockSize)
                      {
                        const int p = tid / (nq1 * nq2);
                        const int q = (tid % (nq1 * nq2)) / nq2;
                        const int r = tid % nq2;

                        T tmp = 0.0;
                        for (unsigned int k = 0; k < nm2; ++k)
                          {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] *
                                   s_basis2[r * nm2 + k];
                          }
                        s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
                      }
                    team_member.team_barrier();

                    // for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2;
                    //      tid += blockSize)
                    //   {
                    //     std::cout << s_wsp1[tid] << " , ";
                    //   }
                    // std::cout << std::endl;

                    // Geometric vals
                    T Grr, Grs, Grt, Gss, Gst, Gtt;
                    T qr, qs, qt;

                    for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2;
                         tid += blockSize)
                      {
                        const int p = tid / (nq1 * nq2);
                        const int q = (tid % (nq1 * nq2)) / nq2;
                        const int r = tid % nq2;

                        qr = 0;
                        qs = 0;
                        qt = 0;

                        // step-5 : Load Geometric Factors, coalesced access
                        Grr =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];
                        Grs =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];
                        Grt =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];
                        Gss =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];
                        Gst =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];
                        Gtt =
                          d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 +
                              p * nq1 * nq2 + q * nq2 + r];

                        // step-6 : Multiply by D
                        for (unsigned int n = 0; n < nq0; n++)
                          {
                            qr += s_wsp1[n * nq1 * nq2 + q * nq2 + r] *
                                  s_dbasis0[n * nq0 + p];
                          }

                        for (unsigned int n = 0; n < nq1; n++)
                          {
                            qs += s_wsp1[p * nq1 * nq2 + n * nq2 + r] *
                                  s_dbasis1[n * nq1 + q];
                          }

                        for (unsigned int n = 0; n < nq2; n++)
                          {
                            qt += s_wsp1[p * nq1 * nq2 + q * nq2 + n] *
                                  s_dbasis2[n * nq2 + r];
                          }


                        // std::cout << qr << " , " << qs << " , " << qt << " ,
                        // "; std::cout << std::endl;

                        // step-7 : Apply chain rule
                        rqr[p * nq1 * nq2 + q * nq2 + r] =
                          Grr * qr + Grs * qs + Grt * qt;
                        rqs[p * nq1 * nq2 + q * nq2 + r] =
                          Grs * qr + Gss * qs + Gst * qt;
                        rqt[p * nq1 * nq2 + q * nq2 + r] =
                          Grt * qr + Gst * qs + Gtt * qt;

                        // std::cout << rqr[p * nq1 * nq2 + q * nq2 + r] << " ,
                        // "
                        //           << rqs[p * nq1 * nq2 + q * nq2 + r] << " ,
                        //           "
                        //           << rqt[p * nq1 * nq2 + q * nq2 + r] << " ,
                        //           ";
                        // std::cout << std::endl;
                      }
                    team_member.team_barrier();

                    // std::cout << std::endl;



                    // step-8 : Compute out vector in GL nodes
                    for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2;
                         tid += blockSize)
                      {
                        const int p = tid / (nq1 * nq2);
                        const int q = (tid % (nq1 * nq2)) / nq2;
                        const int r = tid % nq2;

                        T tmp0 = 0;
                        for (unsigned int n = 0; n < nq0; ++n)
                          tmp0 += rqr[n * nq1 * nq2 + q * nq2 + r] *
                                  s_dbasis0[p * nq0 + n];

                        for (unsigned int n = 0; n < nq1; ++n)
                          tmp0 += rqs[p * nq1 * nq2 + n * nq2 + r] *
                                  s_dbasis1[q * nq1 + n];

                        for (unsigned int n = 0; n < nq2; ++n)
                          tmp0 += rqt[p * nq1 * nq2 + q * nq2 + n] *
                                  s_dbasis2[r * nq2 + n];

                        s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp0;
                      }
                    team_member.team_barrier();

                    // for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2;
                    //      tid += blockSize)
                    //   {
                    //     Kokkos::printf("%f , ", s_wsp1[tid]);
                    //   }
                    // std::cout << std::endl;
                    // team_member.team_barrier();

                    // for (unsigned int i = 0; i < nq0 * nq1 * nq2; ++i)
                    //   std::cout << s_wsp1[i] << " , ";
                    // std::cout << std::endl;



                    /*
                    Interpolate to GLL nodes
                    */

                    // step-9 : direction 2
                    for (unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2;
                         tid += blockSize)
                      {
                        const int q = tid / (nq0 * nm2);
                        const int p = (tid % (nq0 * nm2)) / nm2;
                        const int k = tid % nm2;

                        T tmp = 0.0;
                        for (unsigned int r = 0; r < nq2; ++r)
                          {
                            tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] *
                                   s_basis2[r * nm2 + k];
                          }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                      }
                    team_member.team_barrier();

                    // step-10 : direction 1
                    for (unsigned int tid = threadIdx; tid < nm1 * nm2 * nq0;
                         tid += blockSize)
                      {
                        const int p = tid / (nm1 * nm2);
                        const int j = (tid % (nm1 * nm2)) / nm2;
                        const int k = tid % nm2;

                        T tmp = 0.0;
                        for (unsigned int q = 0; q < nq1; q++)
                          {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] *
                                   s_basis1[q * nm1 + j];
                          }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                      }
                    team_member.team_barrier();

                    // step-11 : direction 0
                    for (unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2;
                         tid += blockSize)
                      {
                        const int i = tid / (nm1 * nm2);
                        const int j = (tid % (nm1 * nm2)) / nm2;
                        const int k = tid % nm2;

                        T tmp = 0.0;
                        for (unsigned int p = 0; p < nq0; ++p)
                          {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] *
                                   s_basis0[p * nm0 + i];
                          }
                        s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
                      }
                    team_member.team_barrier();

                    for (unsigned int i = 0; i < nq0 * nq1 * nq2; ++i)
                      std::cout << s_wsp0[i] << " , ";
                    std::cout << std::endl;

                    // step-12 : Copy wsp0 to out
                    for (unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2;
                         tid += blockSize)
                      {
                        d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
                      }
                    team_member.team_barrier();

                    e += team_member.league_size();
                  }
              });
            Kokkos::fence();
          }
      }
    }



  } // namespace Parallel
} // namespace BK3

#endif // BK3_TEMPLATED_KOKKOS_KERNELS_HPP