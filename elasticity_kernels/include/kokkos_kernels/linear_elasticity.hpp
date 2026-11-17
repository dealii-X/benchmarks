#ifndef LINEAR_ELASTICITY_HPP
#define LINEAR_ELASTICITY_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace LinearElasticity{
namespace Parallel{


template <typename T, const unsigned int nq>
std::vector<double> Kokkos_LinearElasticityOperator(size_t nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const T * basis, const T * dbasis,
    const T*  G, const T*  in, T*  out, const unsigned int ntests,
    const T lambda = (T)1, const T mu = (T)1)
    {   
        const unsigned int nm = nq - 1;
        const unsigned int ncomp = 3;

        if (threadsPerBlock != nelmtPerBatch * nq * nq) {
            throw std::invalid_argument(
                "Kokkos_LinearElasticityOperator requires threadsPerBlock == nelmtPerBatch * nq * nq");
        }

        const size_t ndof_1D      = static_cast<size_t>(nm) * nm * nm;
        const size_t nquad        = static_cast<size_t>(nq) * nq * nq;
        const size_t nbatches     = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch;
        const size_t nelmt_padded = nbatches * nelmtPerBatch;

        T sum = 0.0;
        std::vector<double> results(3);
        {   
            // Capture basis & dbasis by value in Kokkos::Array (CUDA cmem[0] constant bank):
            // compile-time constant indices in unrolled loops fold into FFMA/LDC without LDS/MIO pressure
            Kokkos::Array<T, (nq - 1) * nq> c_basis;
            for(unsigned int i = 0u; i < nm * nq; ++i) c_basis[i] = basis[i];
            Kokkos::Array<T, nq * nq> c_dbasis;
            for(unsigned int i = 0u; i < nq * nq; ++i) c_dbasis[i] = dbasis[i];

            Kokkos::View<const T*, Kokkos::HostSpace> dbasis_view(dbasis, nq * nq);
            Kokkos::View<T*> d_dbasis(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_dbasis"), nq * nq);
            Kokkos::deep_copy(d_dbasis, dbasis_view);

            Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nquad * 10);
            Kokkos::View<T*> d_G(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_G"), nelmt_padded * nquad * 10);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * ncomp * ndof_1D);
            Kokkos::View<T*> d_in(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_in"), nelmt_padded * ncomp * ndof_1D);

            Kokkos::View<T*> d_out(Kokkos::view_alloc(Kokkos::WithoutInitializing, "d_out"), nelmt_padded * ncomp * ndof_1D);

            // Host buffers are not padded, copy only valid entries into nelmt_padded device buffers
            Kokkos::deep_copy(
                Kokkos::subview(d_G, std::make_pair(size_t(0), nelmt * nquad * 10)),
                G_view);

            for (unsigned int c = 0u; c < ncomp; ++c) {
                const size_t dst_start = c * nelmt_padded * ndof_1D;
                const size_t dst_end   = dst_start + nelmt * ndof_1D;
                const size_t src_start = c * nelmt * ndof_1D;
                const size_t src_end   = src_start + nelmt * ndof_1D;
                Kokkos::deep_copy(
                    Kokkos::subview(d_in, std::make_pair(dst_start, dst_end)),
                    Kokkos::subview(in_view, std::make_pair(src_start, src_end)));
            }

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<double>::max();

            //Kokkos with shared memory
            unsigned int ssize = nq * nq + 7 * nelmtPerBatch * nq * nq * nq;          
            
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

                    T r_p0[nq];
                    T r_p1[nq];
                    T r_p2[nq];

                    // Quadrature fluxes of components 1 and 2, computed once and carried in registers
                    T r_f1r[nq], r_f1s[nq], r_f1t[nq];
                    T r_f2r[nq], r_f2s[nq], r_f2t[nq];

                    //shared memory access
                    T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);

                    T *s_dbasis = scratch;

                    T *s_wsp0   = s_dbasis + nq * nq;
                    T *s_wsp1   = s_wsp0   + nelmtPerBatch * nq * nq * nq;

                    T *s_rqr    = s_wsp1   + nelmtPerBatch * nq * nq * nq;
                    T *s_rqs    = s_rqr    + nelmtPerBatch * nq * nq * nq;
                    T *s_rqt    = s_wsp0;

                    T *s_u0     = s_rqs    + nelmtPerBatch * nq * nq * nq;
                    T *s_u1     = s_u0     + nelmtPerBatch * nq * nq * nq;
                    T *s_u2     = s_u1     + nelmtPerBatch * nq * nq * nq;
                    T *s_u[3]   = {s_u0, s_u1, s_u2};

                    const unsigned int threadIdx = team_member.team_rank();
                    const unsigned int blockSize = team_member.team_size();

                    // Copy thread-indexed dbasis to shared memory (basis and uniform dbasis lookups use cmem)
                    for(unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
                    {
                        s_dbasis[tid] = d_dbasis[tid];
                    }
                    team_member.team_barrier();
                        

                    //element batch iteration
                    size_t eb = team_member.league_rank();
                    while(eb < nbatches)
                    {   
                        /*
                        Stage 1: Interpolate to GL nodes (all 3 components required before quadrature coupling)
                        */
                        #pragma unroll
                        for(int c = 0; c < 3; ++c)
                        {
                            const size_t comp_offset  = c * nelmt_padded * nm * nm * nm;
                            const size_t batch_offset = eb * nelmtPerBatch * nm * nm * nm;

                            //step-1 + step-2 : direction 0
                            if(threadIdx < nelmtPerBatch * nm * nm)
                            {
                                int e = threadIdx / (nm * nm);
                                int j = threadIdx % (nm * nm) / nm;
                                int k = threadIdx % nm;
                            
                                #pragma unroll
                                for(int i = 0; i < nm; ++i){
                                    r_p[i] = d_in[comp_offset + batch_offset + e * nm*nm*nm + i * nm*nm + j * nm + k ];
                                }
                            
                                #pragma unroll
                                for (int p = 0; p < nq; ++p) {
                                    T tmp = 0.0;
                                
                                    #pragma unroll
                                    for(int i = 0; i < nm; ++i) {
                                        tmp += c_basis[i * nq + p] * r_p[i];
                                    }
                                
                                    s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                                }
                            }
                            team_member.team_barrier();
                            
                            //step-3 : direction 1
                            if(threadIdx < nelmtPerBatch * nm * nq)
                            {
                                int e = threadIdx / (nq * nm);
                                int p = threadIdx % (nq * nm) / nm;
                                int k = threadIdx % nm;
                                #pragma unroll
                                for(int j = 0; j < nm; ++j){
                                    r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                                }
                            
                                #pragma unroll
                                for (int q = 0; q < nq; ++q) {
                                    T tmp = 0.0;
                                
                                    #pragma unroll
                                    for(int j = 0; j < nm; ++j) {
                                        tmp += c_basis[j * nq + q] * r_q[j];
                                    }
                                
                                    s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                                }
                            }
                            team_member.team_barrier();
                            
                            //step-4 : direction 2 (all blockSize == nelmtPerBatch * nq * nq threads active)
                            {
                                int e = threadIdx / (nq * nq);
                                int q = threadIdx % (nq * nq) / nq;
                                int p = threadIdx % nq;
                            
                                #pragma unroll
                                for(int k = 0; k < nm; ++k){
                                    r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                                }
                                #pragma unroll
                                for (int r = 0; r < nq; ++r) {
                                    T tmp = 0.0;
                                
                                    #pragma unroll
                                    for(int k = 0; k < nm; ++k) {
                                        tmp += c_basis[k * nq + r] * r_r[k];
                                    }
                                
                                    s_u[c][e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
                                }
                            }
                            if (c == 2)
                                team_member.team_barrier();
                        } //interpolate component loop


                        /*
                        Stage 2: Quadrature derivatives, constitutive law, and reference fluxes (evaluated once for all 3 components)
                        */
                        {
                            const int e = threadIdx / (nq * nq);
                            const int q = threadIdx % (nq * nq) / nq;
                            const int r = threadIdx % nq;

                            #pragma unroll
                            for(int n = 0; n < nq; n++)
                            {
                                r_p0[n] = s_u0[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_p1[n] = s_u1[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_p2[n] = s_u2[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_q[n]  = s_dbasis[n * nq + q];
                                r_r[n]  = s_dbasis[n * nq + r];
                            }

                            const size_t g_elem_base = eb * nelmtPerBatch * 10 * nq*nq*nq + e * 10 * nq*nq*nq + q * nq + r;
                            const T two_mu = (T)2 * mu;
                            T wdetJ, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz;
                            T ur0, us0, ut0;
                            T ur1, us1, ut1;
                            T ur2, us2, ut2;

                            #pragma unroll
                            for(int p = 0; p < nq; ++p){

                                ur0 = 0; us0 = 0; ut0 = 0;
                                ur1 = 0; us1 = 0; ut1 = 0;
                                ur2 = 0; us2 = 0; ut2 = 0;

                                //Load Geometric Factors, coalesced access
                                const size_t g_base = g_elem_base + p * nq * nq;
                                wdetJ = d_G[g_base + 0 * nq*nq*nq];
                                drdx  = d_G[g_base + 1 * nq*nq*nq];
                                drdy  = d_G[g_base + 2 * nq*nq*nq];
                                drdz  = d_G[g_base + 3 * nq*nq*nq];
                                dsdx  = d_G[g_base + 4 * nq*nq*nq];
                                dsdy  = d_G[g_base + 5 * nq*nq*nq];
                                dsdz  = d_G[g_base + 6 * nq*nq*nq];
                                dtdx  = d_G[g_base + 7 * nq*nq*nq];
                                dtdy  = d_G[g_base + 8 * nq*nq*nq];
                                dtdz  = d_G[g_base + 9 * nq*nq*nq];

                                // Multiply by D
                                #pragma unroll
                                for(int n = 0; n < nq; n++){
                                    T dp = c_dbasis[n * nq + p];

                                    ur0 += dp * r_p0[n];
                                    us0 += r_q[n] * s_u0[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    ut0 += r_r[n] * s_u0[e * nq*nq*nq + n * nq*nq + q * nq + p];

                                    ur1 += dp * r_p1[n];
                                    us1 += r_q[n] * s_u1[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    ut1 += r_r[n] * s_u1[e * nq*nq*nq + n * nq*nq + q * nq + p];

                                    ur2 += dp * r_p2[n];
                                    us2 += r_q[n] * s_u2[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    ut2 += r_r[n] * s_u2[e * nq*nq*nq + n * nq*nq + q * nq + p];
                                }

                                // Apply chain rule and constitutive law (eps : C : eps)
                                T dudx00 = ur0 * drdx + us0 * dsdx + ut0 * dtdx;
                                T dudx01 = ur0 * drdy + us0 * dsdy + ut0 * dtdy;
                                T dudx02 = ur0 * drdz + us0 * dsdz + ut0 * dtdz;

                                T dudx10 = ur1 * drdx + us1 * dsdx + ut1 * dtdx;
                                T dudx11 = ur1 * drdy + us1 * dsdy + ut1 * dtdy;
                                T dudx12 = ur1 * drdz + us1 * dsdz + ut1 * dtdz;

                                T dudx20 = ur2 * drdx + us2 * dsdx + ut2 * dtdx;
                                T dudx21 = ur2 * drdy + us2 * dsdy + ut2 * dtdy;
                                T dudx22 = ur2 * drdz + us2 * dsdz + ut2 * dtdz;

                                T div_u = dudx00 + dudx11 + dudx22;

                                const T sigma_xx = lambda * div_u + two_mu * dudx00;
                                const T sigma_yy = lambda * div_u + two_mu * dudx11;
                                const T sigma_zz = lambda * div_u + two_mu * dudx22;
                                const T sigma_xy = mu * (dudx01 + dudx10);
                                const T sigma_xz = mu * (dudx02 + dudx20);
                                const T sigma_yz = mu * (dudx12 + dudx21);

                                // Component 0 directly into shared memory for step-8
                                s_rqr[e * nq*nq*nq + p * nq * nq + q * nq + r] = wdetJ * (sigma_xx * drdx + sigma_xy * drdy + sigma_xz * drdz);
                                s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = wdetJ * (sigma_xx * dsdx + sigma_xy * dsdy + sigma_xz * dsdz);
                                s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = wdetJ * (sigma_xx * dtdx + sigma_xy * dtdy + sigma_xz * dtdz);

                                // Components 1 and 2 kept in registers for subsequent passes
                                r_f1r[p] = wdetJ * (sigma_xy * drdx + sigma_yy * drdy + sigma_yz * drdz);
                                r_f1s[p] = wdetJ * (sigma_xy * dsdx + sigma_yy * dsdy + sigma_yz * dsdz);
                                r_f1t[p] = wdetJ * (sigma_xy * dtdx + sigma_yy * dtdy + sigma_yz * dtdz);

                                r_f2r[p] = wdetJ * (sigma_xz * drdx + sigma_yz * drdy + sigma_zz * drdz);
                                r_f2s[p] = wdetJ * (sigma_xz * dsdx + sigma_yz * dsdy + sigma_zz * dsdz);
                                r_f2t[p] = wdetJ * (sigma_xz * dtdx + sigma_yz * dtdy + sigma_zz * dtdz);
                            }
                        }
                        team_member.team_barrier();


                        /*
                        Stage 3: Adjoint differentiation & projection to GLL nodes for each component
                        */
                        #pragma unroll
                        for(int c = 0; c < 3; ++c)
                        {
                            const size_t comp_offset  = c * nelmt_padded * nm * nm * nm;
                            const size_t batch_offset = eb * nelmtPerBatch * nm * nm * nm;

                            // c == 0 fluxes are already in shared memory from Stage 2
                            if (c > 0)
                            {
                                const int e = threadIdx / (nq * nq);
                                const int q = threadIdx % (nq * nq) / nq;
                                const int r = threadIdx % nq;

                                #pragma unroll
                                for(int p = 0; p < nq; ++p){
                                    // r-fluxes (r_f1r / r_f2r) stay in registers for step-8 directly
                                    s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = (c == 1) ? r_f1s[p] : r_f2s[p];
                                    s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = (c == 1) ? r_f1t[p] : r_f2t[p];
                                }
                                team_member.team_barrier();
                            }

                            //step-8 : Compute out vector in GL nodes
                            {
                                int e = threadIdx / (nq * nq);
                                int q = threadIdx % (nq * nq) / nq;
                                int r = threadIdx % nq;
                            
                                //copy to register
                                #pragma unroll
                                for(int n = 0; n < nq; n++)
                                {
                                    if (c == 0)
                                        r_p[n] = s_rqr[e * nq*nq*nq + n * nq * nq + q * nq + r];
                                    r_q[n] = s_dbasis[q * nq + n];
                                    r_r[n] = s_dbasis[r * nq + n];
                                }
                            
                                #pragma unroll
                                for(int p = 0; p < nq; ++p)
                                {
                                    T tmp0 = 0;
                                    #pragma unroll
                                    for(int n = 0; n < nq; ++n) {
                                        const T rp_n = (c == 0) ? r_p[n] : ((c == 1) ? r_f1r[n] : r_f2r[n]);
                                        tmp0 += rp_n * c_dbasis[p * nq + n];
                                    }
                                
                                    #pragma unroll
                                    for(int n = 0; n < nq; ++n)                
                                        tmp0 += s_rqs[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                                
                                    #pragma unroll
                                    for(int n = 0; n < nq; ++n)
                                        tmp0 += s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                                
                                    s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
                                }
                            }
                            team_member.team_barrier();


                            /*
                            Interpolate to GLL nodes
                            */
                        
                            //step-9 : direction 2
                            {                
                                int e = threadIdx / (nq * nq);
                                int q = threadIdx % (nq * nq) / nq;
                                int p = threadIdx % nq;
                            
                                #pragma unroll
                                for(int r = 0; r < nq; ++r){
                                    r_r[r] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p];
                                }
                            
                                #pragma unroll
                                for (int k = 0; k < nm; ++k) {
                                    T tmp = 0.0;
                                
                                    #pragma unroll
                                    for(int r = 0; r < nq; ++r) {
                                        tmp += c_basis[k * nq + r] * r_r[r];
                                    }
                                
                                    s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                                }   
                            }
                            team_member.team_barrier();

                            //step-10 : direction 1
                            if(threadIdx < nelmtPerBatch * nm * nq)
                            {   
                                int e = threadIdx / (nm * nq);
                                int k = threadIdx % (nm * nq) / nq;
                                int p = threadIdx % nq;
                            
                                #pragma unroll
                                for(int q = 0; q < nq; ++q){
                                    r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                                }
                            
                                #pragma unroll
                                for (int j = 0; j < nm; ++j) {
                                    T tmp = 0.0;
                                
                                    #pragma unroll
                                    for(int q = 0; q < nq; ++q) {
                                        tmp += c_basis[j * nq + q] * r_q[q];
                                    }
                                    s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                                }
                            }
                            team_member.team_barrier();

                            //step-11 + step 12 : direction 0 and write d_out
                            if(threadIdx < nelmtPerBatch * nm * nm)
                            {   
                                int e = threadIdx / (nm * nm);
                                int j = threadIdx % (nm * nm) / nm;
                                int k = threadIdx % nm;
                                const size_t out_base = comp_offset + batch_offset + e * nm*nm*nm + j * nm + k;
                            
                                #pragma unroll
                                for(int p = 0; p < nq; ++p){
                                    r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                                }
                            
                                #pragma unroll
                                for (int i = 0; i < nm; ++i) {
                                    T tmp = 0.0;
                                    #pragma unroll
                                    for(int p = 0; p < nq; ++p) {
                                        tmp += c_basis[i * nq + p] * r_p[p];
                                    }

                                    d_out[out_base + i * nm*nm] = tmp;
                                }
                            }
                            if (c == 2 && eb + team_member.league_size() < nbatches)
                                team_member.team_barrier();
                        } //component loop

                        eb += team_member.league_size();
                    }
                });
            Kokkos::fence();

            kokkosTimer.stop();
            const double t_w = kokkosTimer.elapsedSeconds();
            time_kokkos     = std::min(time_kokkos, t_w);
        }

        const size_t comp_stride_valid  = nelmt * ndof_1D;
        const size_t comp_stride_padded = nelmt_padded * ndof_1D;
        Kokkos::parallel_reduce(nelmt * ncomp * ndof_1D,
            KOKKOS_LAMBDA(size_t i, T &val) {
                const size_t c   = i / comp_stride_valid;
                const size_t rem = i % comp_stride_valid;
                const T v = d_out(c * comp_stride_padded + rem);
                val += v * v;
            },
            sum);

        double gdofPerSeconds = 1.0e-9 * nelmt * ncomp * nm * nm * nm / time_kokkos;
        results[0] = gdofPerSeconds; 
        results[1] = sum;
        results[2] = time_kokkos;
    }

    return results;
}


} //namespace Parallel
} //namespace LinearElasticity

#endif //LINEAR_ELASTICITY_HPP
