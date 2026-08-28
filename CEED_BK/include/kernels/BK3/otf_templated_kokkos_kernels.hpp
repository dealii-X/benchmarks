#ifndef OTF_BK3_TEMPLATED_KOKKOS_KERNELS_HPP
#define OTF_BK3_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <limits>
#include <algorithm>

namespace BK3{
namespace Parallel{

/*
 * On-the-fly BK3 Laplace operator.
 *
 * Compared with the original kernel:
 *   - G is not stored or loaded.
 *   - The input is the physical coordinate field at the GL quadrature points.
 *   - At every quadrature point we build J, its cofactor matrix C, and det(J).
 *   - The geometric action is applied as
 *         (w/detJ) C (C^T g)
 *     without ever forming G = (w/detJ) C C^T.
 *   - The existing sum-factorization / team / scratch structure is otherwise
 *     kept intact.
 *
 * coord_q layout:
 *   coord_q[((e*3 + c)*q^3) + p*q*q + q*nq + r]
 *   c = 0 -> x, c = 1 -> y, c = 2 -> z.
 *
 * weights layout:
 *   weights[p], p = 0,...,q-1.
 *
 * IMPORTANT:
 *   coord_q is assumed to contain the physical coordinates evaluated at the
 *   GL quadrature points. If the only available coordinates are the GLL
 *   nodal coordinates, an additional tensor-product interpolation stage is
 *   required to construct coord_q.
 */
template <typename T, const unsigned int nq>
std::vector<double> Kokkos_LaplaceOperator_OTF(
    const unsigned int nelmt,
    const unsigned int nelmtPerBatch,
    const unsigned int numBlocks,
    const unsigned int threadsPerBlock,
    const T *__restrict__ basis,
    const T *__restrict__ dbasis,
    const T *__restrict__ weights,
    const T *__restrict__ coord_q,
    const T *__restrict__ in,
    T* __restrict__ out,
    const unsigned int ntests)
{
    const unsigned int nm = nq - 1;

    T sum = 0.0;
    std::vector<double> results(3);

    {
        Kokkos::View<const T*, Kokkos::HostSpace> basis_view(basis, nm * nq);
        Kokkos::View<T*> d_basis("d_basis", nm * nq);
        Kokkos::deep_copy(d_basis, basis_view);

        Kokkos::View<const T*, Kokkos::HostSpace> dbasis_view(dbasis, nq * nq);
        Kokkos::View<T*> d_dbasis("d_dbasis", nq * nq);
        Kokkos::deep_copy(d_dbasis, dbasis_view);

        Kokkos::View<const T*, Kokkos::HostSpace> weights_view(weights, nq);
        Kokkos::View<T*> d_weights("d_weights", nq);
        Kokkos::deep_copy(d_weights, weights_view);

        // coord_q is [element][x/y/z][p][q][r], so there are 3*q^3 values/element.
        Kokkos::View<const T*, Kokkos::HostSpace> coord_view(
            coord_q, nelmt * 3 * nq * nq * nq);
        Kokkos::View<T*> d_coord("d_coord", nelmt * 3 * nq * nq * nq);
        Kokkos::deep_copy(d_coord, coord_view);

        Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * nm * nm * nm);
        Kokkos::View<T*> d_in("d_in", nelmt * nm * nm * nm);
        Kokkos::deep_copy(d_in, in_view);

        Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * nm * nm * nm);
        Kokkos::View<T*> d_out("d_out", nelmt * nm * nm * nm);

        Timer kokkosTimer;
        double time_kokkos = std::numeric_limits<T>::max();

        // Keep the original scratch layout.  No extra q^3 geometry arrays are
        // introduced: coordinates remain in d_coord, while C is register-local.
        const unsigned int ssize = nm * nq + nq * nq
                                  + 4 * nelmtPerBatch * nq * nq * nq;
        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            kokkosTimer.start();
            Kokkos::parallel_for(policy,
                KOKKOS_LAMBDA (member_type team_member){

                T r_p[nq];
                T r_q[nq];
                T r_r[nq];

                T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);

                T *s_basis = scratch;
                T *s_dbasis = s_basis + nq * nm;

                T *s_wsp0 = s_dbasis + nq * nq;
                T *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;

                T *s_rqr = s_wsp1 + nelmtPerBatch * nq * nq * nq;
                T *s_rqs = s_rqr  + nelmtPerBatch * nq * nq * nq;
                T *s_rqt = s_wsp0; // same alias as original kernel

                const unsigned int threadIdx = team_member.team_rank();
                const unsigned int blockSize = team_member.team_size();

                // Copy 1D data to shared memory.
                for (unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
                    s_basis[tid] = d_basis[tid];

                for (unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
                    s_dbasis[tid] = d_dbasis[tid];

                team_member.team_barrier();

                // element batch iteration
                int eb = team_member.league_rank();
                while (eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
                {
                    const int c_nelmtPerBatch =
                        std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

                    // ----------------------------------------------------------
                    // step-1 : Copy from input to wsp0
                    // ----------------------------------------------------------
                    for (int idx = threadIdx;
                         idx < c_nelmtPerBatch * nm * nm * nm;
                         idx += blockSize)
                    {
                        s_wsp0[idx] = d_in[
                            eb * nelmtPerBatch * nm * nm * nm + idx];
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-2 : direction 0
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nm * nm;
                         tid += blockSize)
                    {
                        const int e = tid / (nm * nm);
                        const int j = tid % (nm * nm) / nm;
                        const int k = tid % nm;

                        for (int i = 0; i < nm; ++i)
                            r_p[i] = s_wsp0[e * nm * nm * nm
                                           + i * nm * nm + j * nm + k];

                        for (int p = 0; p < nq; ++p)
                        {
                            T tmp = 0.0;
                            for (int i = 0; i < nm; ++i)
                                tmp += s_basis[i * nq + p] * r_p[i];

                            s_wsp1[e * nq * nm * nm
                                 + p * nm * nm + j * nm + k] = tmp;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-3 : direction 1
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nm * nq;
                         tid += blockSize)
                    {
                        const int e = tid / (nq * nm);
                        const int p = tid % (nq * nm) / nm;
                        const int k = tid % nm;

                        for (int j = 0; j < nm; ++j)
                            r_q[j] = s_wsp1[e * nq * nm * nm
                                           + p * nm * nm + j * nm + k];

                        for (int q = 0; q < nq; ++q)
                        {
                            T tmp = 0.0;
                            for (int j = 0; j < nm; ++j)
                                tmp += s_basis[j * nq + q] * r_q[j];

                            s_wsp0[e * nq * nq * nm
                                 + q * nq * nm + p * nm + k] = tmp;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-4 : direction 2
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nq * nq;
                         tid += blockSize)
                    {
                        const int e = tid / (nq * nq);
                        const int q = tid % (nq * nq) / nq;
                        const int p = tid % nq;

                        for (int k = 0; k < nm; ++k)
                            r_r[k] = s_wsp0[e * nq * nq * nm
                                           + q * nq * nm + p * nm + k];

                        for (int r = 0; r < nq; ++r)
                        {
                            T tmp = 0.0;
                            for (int k = 0; k < nm; ++k)
                                tmp += s_basis[k * nq + r] * r_r[k];

                            s_wsp1[e * nq * nq * nq
                                 + r * nq * nq + q * nq + p] = tmp;
                        }
                    }
                    team_member.team_barrier();

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
                    for (int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                    {
                        const int e = tid / (nq * nq);
                        const int q = tid % (nq * nq) / nq;
                        const int r = tid % nq;

                        for (int n = 0; n < nq; ++n)
                        {
                            r_p[n] = s_wsp1[e * nq * nq * nq + r * nq * nq + q * nq + n];
                            r_q[n] = s_dbasis[n * nq + q];
                            r_r[n] = s_dbasis[n * nq + r];
                        }

                        for (int p = 0; p < nq; ++p)
                        {
                            // --------------------------------------------------
                            // 5a. Reference-space gradient of u.
                            // Keep the original qr/qs/qt definitions so the new
                            // kernel has the same component ordering as the old
                            // G-action.
                            // --------------------------------------------------
                            T qr = 0.0;
                            T qs = 0.0;
                            T qt = 0.0;

                            for (int n = 0; n < nq; ++n)
                            {
                                qr += s_dbasis[n * nq + p] * r_p[n];
                                qs += r_q[n] * s_wsp1[
                                    e * nq * nq * nq + r * nq * nq + n * nq + p];
                                qt += r_r[n] * s_wsp1[
                                    e * nq * nq * nq + n * nq * nq + q * nq + p];
                            }

                            // --------------------------------------------------
                            // 5b. Read the physical coordinates at this
                            // quadrature line and build J.
                            //
                            // coord layout per element:
                            //   x[p,q,r], y[p,q,r], z[p,q,r]
                            //
                            // The nine entries are generated directly from the
                            // q-point coordinate field, using the same dbasis
                            // lines already held in registers/shared memory.
                            // --------------------------------------------------

                            const unsigned int coord_base = (eb * nelmtPerBatch + e) * 3u * nq * nq * nq;
                            const unsigned int xbase = coord_base;
                            const unsigned int ybase = coord_base + nq * nq * nq;
                            const unsigned int zbase = coord_base + 2u * nq * nq * nq;

                            T J00 = 0.0, J01 = 0.0, J02 = 0.0;
                            T J10 = 0.0, J11 = 0.0, J12 = 0.0;
                            T J20 = 0.0, J21 = 0.0, J22 = 0.0;

                            for (int n = 0; n < nq; ++n)
                            {
                                // x derivatives
                                const T x_r = d_coord[xbase + n * nq * nq + q * nq + r];
                                const T x_s = d_coord[xbase + p * nq * nq + n * nq + r];
                                const T x_t = d_coord[xbase + p * nq * nq + q * nq + n];

                                J00 += s_dbasis[n * nq + p] * x_r;
                                J01 += r_q[n] * x_s;
                                J02 += r_r[n] * x_t;

                                // y derivatives
                                const T y_r = d_coord[ybase + n * nq * nq + q * nq + r];
                                const T y_s = d_coord[ybase + p * nq * nq + n * nq + r];
                                const T y_t = d_coord[ybase + p * nq * nq + q * nq + n];

                                J10 += s_dbasis[n * nq + p] * y_r;
                                J11 += r_q[n] * y_s;
                                J12 += r_r[n] * y_t;

                                // z derivatives
                                const T z_r = d_coord[zbase + n * nq * nq + q * nq + r];
                                const T z_s = d_coord[zbase + p * nq * nq + n * nq + r];
                                const T z_t = d_coord[zbase + p * nq * nq + q * nq + n];

                                J20 += s_dbasis[n * nq + p] * z_r;
                                J21 += r_q[n] * z_s;
                                J22 += r_r[n] * z_t;
                            }

                            // --------------------------------------------------
                            // 5c. Cofactor matrix C = det(J) J^{-T}
                            // --------------------------------------------------
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
                            const T scale = (d_weights[p] * d_weights[q] * d_weights[r]) / detJ;

                            // --------------------------------------------------
                            // 5d. Direct factored action C (C^T g).
                            //
                            // Preserve the component ordering of the original
                            // code exactly: its metric multiply uses
                            //   [qt, qs, qr]^T
                            // as the input vector to G, then stores the three
                            // outputs into [rqr,rqs,rqt].
                            // --------------------------------------------------
                            const T t0 = C00 * qt + C10 * qs + C20 * qr;
                            const T t1 = C01 * qt + C11 * qs + C21 * qr;
                            const T t2 = C02 * qt + C12 * qs + C22 * qr;

                            s_rqr[e * nq * nq * nq + p * nq * nq + q * nq + r] = scale * (C00 * t0 + C01 * t1 + C02 * t2);
                            s_rqs[e * nq * nq * nq + p * nq * nq + q * nq + r] = scale * (C10 * t0 + C11 * t1 + C12 * t2);
                            s_rqt[e * nq * nq * nq + p * nq * nq + q * nq + r] = scale * (C20 * t0 + C21 * t1 + C22 * t2);
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-6 : transpose direction 2
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nq * nq;
                         tid += blockSize)
                    {
                        const int e = tid / (nq * nq);
                        const int q = tid % (nq * nq) / nq;
                        const int r = tid % nq;

                        for (int n = 0; n < nq; ++n)
                        {
                            r_p[n] = s_rqr[e * nq * nq * nq
                                          + n * nq * nq + q * nq + r];
                            r_q[n] = s_dbasis[q * nq + n];
                            r_r[n] = s_dbasis[r * nq + n];
                        }

                        for (int p = 0; p < nq; ++p)
                        {
                            T tmp0 = 0.0;

                            for (int n = 0; n < nq; ++n)
                                tmp0 += r_p[n] * s_dbasis[p * nq + n];

                            for (int n = 0; n < nq; ++n)
                                tmp0 += s_rqs[e * nq * nq * nq
                                             + p * nq * nq + n * nq + r] * r_q[n];

                            for (int n = 0; n < nq; ++n)
                                tmp0 += s_rqt[e * nq * nq * nq
                                             + p * nq * nq + q * nq + n] * r_r[n];

                            s_wsp1[e * nq * nq * nq
                                 + r * nq * nq + q * nq + p] = tmp0;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-7 : Interpolate to GLL, direction 2
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nq * nq;
                         tid += blockSize)
                    {
                        const int e = tid / (nq * nq);
                        const int q = tid % (nq * nq) / nq;
                        const int p = tid % nq;

                        for (int r = 0; r < nq; ++r)
                            r_r[r] = s_wsp1[e * nq * nq * nq
                                           + r * nq * nq + q * nq + p];

                        for (int k = 0; k < nm; ++k)
                        {
                            T tmp = 0.0;
                            for (int r = 0; r < nq; ++r)
                                tmp += s_basis[k * nq + r] * r_r[r];

                            s_wsp0[e * nm * nq * nq
                                 + k * nq * nq + q * nq + p] = tmp;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-8 : direction 1
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nm * nq;
                         tid += blockSize)
                    {
                        const int e = tid / (nm * nq);
                        const int k = tid % (nm * nq) / nq;
                        const int p = tid % nq;

                        for (int q = 0; q < nq; ++q)
                            r_q[q] = s_wsp0[e * nm * nq * nq
                                           + k * nq * nq + q * nq + p];

                        for (int j = 0; j < nm; ++j)
                        {
                            T tmp = 0.0;
                            for (int q = 0; q < nq; ++q)
                                tmp += s_basis[j * nq + q] * r_q[q];

                            s_wsp1[e * nm * nm * nq
                                 + k * nm * nq + j * nq + p] = tmp;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-9 : direction 0
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nm * nm;
                         tid += blockSize)
                    {
                        const int e = tid / (nm * nm);
                        const int j = tid % (nm * nm) / nm;
                        const int k = tid % nm;

                        for (int p = 0; p < nq; ++p)
                            r_p[p] = s_wsp1[e * nm * nm * nq
                                          + k * nm * nq + j * nq + p];

                        for (int i = 0; i < nm; ++i)
                        {
                            T tmp = 0.0;
                            for (int p = 0; p < nq; ++p)
                                tmp += s_basis[i * nq + p] * r_p[p];

                            s_wsp0[e * nm * nm * nm
                                 + i * nm * nm + j * nm + k] = tmp;
                        }
                    }
                    team_member.team_barrier();

                    // ----------------------------------------------------------
                    // step-10 : Copy wsp0 to out
                    // ----------------------------------------------------------
                    for (int tid = threadIdx;
                         tid < c_nelmtPerBatch * nm * nm * nm;
                         tid += blockSize)
                    {
                        d_out[eb * nelmtPerBatch * nm * nm * nm + tid] = s_wsp0[tid];
                    }
                    team_member.team_barrier();

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
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            sum);

        T gdofPerSeconds =
            1.0e-9 * nelmt * nm * nm * nm / time_kokkos;

        results[0] = gdofPerSeconds;
        results[1] = sum;
        results[2] = time_kokkos;
    }

    return results;
}

} // namespace Parallel
} // namespace BK3

#endif // OTF_BK3_TEMPLATED_KOKKOS_KERNELS_HPP