#ifndef OTF2_BK3_TEMPLATED_KOKKOS_KERNELS_HPP
#define OTF2_BK3_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <limits>
#include <algorithm>

namespace BK3 {
namespace Parallel {

template <typename T, const unsigned int nq>
std::vector<double> Kokkos_LaplaceOperator_OTF2(
    const size_t nelmt,
    const unsigned int numBlocks,
    const unsigned int threadsPerBlock,
    Kokkos::View<T**> d_basis,
    Kokkos::View<T**> d_dbasis,
    Kokkos::View<T*>  d_weights,
    Kokkos::View<T*****> d_coord,
    Kokkos::View<T****> d_in,
    Kokkos::View<T****> d_out,
    const unsigned int ntests)
{
    const unsigned int nm = nq - 1;
    const unsigned int nelmtPerBlock = threadsPerBlock;
    T sum = 0.0;
    std::vector<double> results(3);

    {
        Timer kokkosTimer;
        double time_kokkos = std::numeric_limits<T>::max();

        const unsigned int ssize = nm * nq + nq * nq;
        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            kokkosTimer.start();
            Kokkos::parallel_for(policy,
                KOKKOS_LAMBDA (member_type team_member) {

                T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
                T *s_basis = scratch;
                T *s_dbasis = s_basis + nq * nm;
             
                const unsigned int threadIdx = team_member.team_rank();
                const unsigned int blockSize = team_member.team_size();

                for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize) {
                    s_basis[tid] = d_basis(tid / nq, tid % nq);
                }

                for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize) {
                    s_dbasis[tid] = d_dbasis(tid / nq, tid % nq);
                }

                team_member.team_barrier();

                T r_basis[nq*nm];
                T r_dbasis[nq*nq];
                T r_weights[nq];


                for (unsigned int i = 0; i < nm * nq; ++i)
                {
                    r_basis[i] = s_basis[i];
                }

                for (unsigned int i = 0; i < nq * nq; ++i)
                {
                    r_dbasis[i] = s_dbasis[i];
                }

                for (unsigned int i = 0; i < nq; ++i) {
                    r_weights[i] = d_weights(i);
                }



                // Element batch iteration
                size_t eb = team_member.league_rank();
                const size_t totalBatches = (nelmt + nelmtPerBlock - 1) / nelmtPerBlock;

                while (eb < totalBatches)
                {
                    size_t e = eb * nelmtPerBlock + threadIdx;

                    T r_wsp0[nq][nq][nq] = {};
                    T r_wsp1[nq][nq][nq] = {};

                    // ----------------------------------------------------------
                    // Step-1 : Copy from input to register
                    // ----------------------------------------------------------
            
                    for (unsigned int i = 0; i < nm; ++i) {
                        for (unsigned int j = 0; j < nm; ++j) {
                            for (unsigned int k = 0; k < nm; ++k) {
                                unsigned int idx = i * nm * nm + j * nm + k;
                                    r_wsp0[i][j][k] = d_in(e, i, j, k);
                            }

                        }
                    }

                    // ----------------------------------------------------------
                    // step-2 : direction 0
                    // ----------------------------------------------------------

                    for (unsigned int p = 0; p < nq; ++p)
                    {
                        for (unsigned int j = 0; j < nm; ++j)
                        {
                            for (unsigned int k = 0; k < nm; ++k)
                            {
                                T tmp = 0.0;
                                for (unsigned int i = 0; i < nm; ++i)
                                {
                                    tmp += r_basis[i * nq + p] * r_wsp0[i][j][k];
                                }
                                r_wsp1[p][j][k] = tmp;
                            }
                        }
                    }

                    // ----------------------------------------------------------
                    // step-3 : direction 1
                    // ----------------------------------------------------------

                    for (unsigned int q = 0; q < nq; ++q)
                    {
                        for (unsigned int p = 0; p < nq; ++p)
                        {
                            for (unsigned int k = 0; k < nm; ++k)
                            {
                                T tmp = 0.0;
                                for (unsigned int j = 0; j < nm; ++j)
                                {
                                    tmp += r_basis[j * nq + q] * r_wsp1[p][j][k];
                                }
                                r_wsp0[q][p][k] = tmp;
                            }
                        }
                    }

                    // ----------------------------------------------------------
                    // step-4 : direction 2
                    // ----------------------------------------------------------

                    for (unsigned int r = 0; r < nq; ++r)
                    {
                        for (unsigned int q = 0; q < nq; ++q)
                        {
                            for (unsigned int p = 0; p < nq; ++p)
                            {
                                T tmp = 0.0;
                                for (unsigned int k = 0; k < nm; ++k)
                                {
                                    tmp += r_basis[k * nq + r] * r_wsp0[q][p][k];
                                }
                                r_wsp1[r][q][p] = tmp;
                            }
                        }
                    }

                    for (unsigned int r = 0; r < nq; ++r)
                    {
                        for (unsigned int q = 0; q < nq; ++q)
                        {
                            for (unsigned int p = 0; p < nq; ++p)
                            {
                                r_wsp0[r][q][p] = 0.0;
                            }
                        }
                    }


                    // ----------------------------------------------------------
                    // step-5 : metric action, ON THE FLY
                    //
                    // Original:
                    //   load six entries of symmetric G
                    //   [r] = G [qr,qs,qt]  (with the same component ordering
                    //                         as the original chain-rule code)
                    //
                    // New:
                    //   J -> C,detJ
                    //   t = C^T g
                    //   r = (w/detJ) C t
                    //
                    // C is never stored in global/shared memory.
                    // It exists only as scalar register temporaries.
                    // ----------------------------------------------------------
                            
                    for (unsigned int r = 0; r < nq; ++r)
                    {
                        for (unsigned int q = 0; q < nq; ++q)
                        {
                            for (unsigned int p = 0; p < nq; ++p)
                            {
                                T qr = 0.0, qs = 0.0, qt = 0.0;

                                T J00 = 0.0, J01 = 0.0, J02 = 0.0;
                                T J10 = 0.0, J11 = 0.0, J12 = 0.0;
                                T J20 = 0.0, J21 = 0.0, J22 = 0.0;
                                
                                // 1. & 2. Compute Jacobian and Gradients
                                for (unsigned int n = 0; n < nq; ++n)
                                {
                                    // Gradients
                                    qr += r_dbasis[n * nq + p] * r_wsp1[r][q][n];
                                    qs += r_dbasis[n * nq + q] * r_wsp1[r][n][p];
                                    qt += r_dbasis[n * nq + r] * r_wsp1[n][q][p];
                                    
                                    // Jacobian
                                    J00 += r_dbasis[n * nq + p] * d_coord(e, 0, n, q, r);
                                    J10 += r_dbasis[n * nq + p] * d_coord(e, 0, p, n, r);
                                    J20 += r_dbasis[n * nq + p] * d_coord(e, 0, p, q, n);
                                    
                                    J01 += r_dbasis[n * nq + q] * d_coord(e, 0, n, q, r);
                                    J11 += r_dbasis[n * nq + q] * d_coord(e, 0, p, n, r);
                                    J21 += r_dbasis[n * nq + q] * d_coord(e, 0, p, q, n);
                                    
                                    J02 += r_dbasis[n * nq + r] * d_coord(e, 0, n, q, r);
                                    J12 += r_dbasis[n * nq + r] * d_coord(e, 0, p, n, r);
                                    J22 += r_dbasis[n * nq + r] * d_coord(e, 0, p, q, n);
                                }
                                
                                // 3. Cofactor matrix C = det(J) J^{-T}
                                const T C00 = J11 * J22 - J12 * J21;
                                const T C01 = J02 * J21 - J01 * J22;
                                const T C02 = J01 * J12 - J02 * J11;
                                
                                const T C10 = J12 * J20 - J10 * J22;
                                const T C11 = J00 * J22 - J02 * J20;
                                const T C12 = J02 * J10 - J00 * J12;
                                
                                const T C20 = J10 * J21 - J11 * J20;
                                const T C21 = J01 * J20 - J00 * J21;
                                const T C22 = J00 * J11 - J01 * J10;
                                
                                const T detJ = J00 * C00 + J01 * C10 + J02 * C20;

                                const T scale = (r_weights[p] * r_weights[q] * r_weights[r]) / detJ;
                                
                                const T t0 = C00 * qt + C10 * qs + C20 * qr;
                                const T t1 = C01 * qt + C11 * qs + C21 * qr;
                                const T t2 = C02 * qt + C12 * qs + C22 * qr;
                                
                                const T rqr_val = scale * (C00 * t0 + C01 * t1 + C02 * t2);
                                const T rqs_val = scale * (C10 * t0 + C11 * t1 + C12 * t2);
                                const T rqt_val = scale * (C20 * t0 + C21 * t1 + C22 * t2);
                                
                                for (unsigned int n = 0; n < nq; ++n)
                                {
                                    r_wsp0[r][q][n] += rqr_val * r_dbasis[n * nq + p];
                                    r_wsp0[r][n][p] += rqs_val * r_dbasis[n * nq + q];
                                    r_wsp0[n][q][p] += rqt_val * r_dbasis[n * nq + r];
                                }
                            }
                        }
                    }
                    
                    // ----------------------------------------------------------
                    // step-7 : Interpolate to GLL, direction 2
                    // ----------------------------------------------------------

                    for (unsigned int k = 0; k < nm; ++k)
                    {
                        for (unsigned int q = 0; q < nq; ++q)
                        {
                            for (unsigned int p = 0; p < nq; ++p)
                            {
                                for (unsigned int r = 0; r < nq; ++r)
                                {
                                    r_wsp1[k][q][p] += r_basis[k * nq + r] * r_wsp0[r][q][p];
                                }
                            }
                        }
                    }

                    // ----------------------------------------------------------
                    // step-8 : direction 1
                    // ----------------------------------------------------------

                    for (unsigned int k = 0; k < nm; ++k)
                    {
                        for (unsigned int j = 0; j < nm; ++j)
                        {
                            for (unsigned int p = 0; p < nq; ++p)
                            {
                                T tmp = 0.0;
                                for (unsigned int q = 0; q < nq; ++q)
                                {
                                    tmp += r_basis[j * nq + q] * r_wsp1[k][q][p];
                                }
                                r_wsp0[k][j][p] = tmp;
                            }
                        }
                    }

                    // ----------------------------------------------------------
                    // step-9 : direction 0
                    // ----------------------------------------------------------

                    for (unsigned int i = 0; i < nm; ++i)
                    {
                        for (unsigned int j = 0; j < nm; ++j)
                        {
                            for (unsigned int k = 0; k < nm; ++k)
                            {
                                T tmp = 0.0;
                                for (unsigned int p = 0; p < nq; ++p)
                                {
                                    tmp += r_basis[i * nq + p] * r_wsp0[k][j][p];
                                }
                                r_wsp1[i][j][k] = tmp;
                            }
                        }
                    }

                    // ----------------------------------------------------------
                    // step-10 : Copy wsp0 to out
                    // ----------------------------------------------------------
                    for (unsigned int i = 0; i < nm; ++i)
                    {
                        for (unsigned int j = 0; j < nm; ++j)
                        {
                            for (unsigned int k = 0; k < nm; ++k)
                            {     
                                d_out(e, i, j, k) = r_wsp1[i][j][k];
                            }
                        }
                    }
                    
                    eb += team_member.league_size();
                }
            });
            Kokkos::fence();

            kokkosTimer.stop();
            const double t_w = kokkosTimer.elapsedSeconds();
            time_kokkos = std::min(time_kokkos, t_w);
        }

        Kokkos::parallel_reduce(
            nelmt * nm * nm * nm,
            KOKKOS_LAMBDA(const size_t idx, T &val) {

            const size_t e = idx / (nm * nm * nm);
            const size_t i = (idx / (nm * nm)) % nm;
            const size_t j = (idx / nm) % nm;
            const size_t k = idx % nm;

            const T value = d_out(e, i, j, k);
            val += value * value;
        },  sum);

        T gdofPerSeconds = 1.0e-9 * nelmt * nm * nm * nm / time_kokkos;

        results[0] = gdofPerSeconds;
        results[1] = sum;
        results[2] = time_kokkos;
    }

    return results;
}

} // namespace Parallel
} // namespace BK3

#endif // OTF2_BK3_TEMPLATED_KOKKOS_KERNELS_HPP