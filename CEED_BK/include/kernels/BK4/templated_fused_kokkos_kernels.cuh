#ifndef BK4_TEMPLATED_FUSED_KOKKOS_KERNELS_HPP
#define BK4_TEMPLATED_FUSED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <limits>
#include <algorithm>

namespace BK4{
namespace Parallel{


template <typename T, const unsigned int nq>
std::vector<double> Kokkos_LaplaceOperator(size_t nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const T * basis, const T * dbasis,
    const T*  G, const T*  in, T*  out, const unsigned int ntests)
    {   
        const unsigned int nm = nq - 1;
        constexpr unsigned int ncomp = 3;

        T sum = 0.0;
        std::vector<double> results(3);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> basis_view(basis, nm * nq);
            Kokkos::View<T*> d_basis("d_basis", nm * nq);
            Kokkos::deep_copy(d_basis, basis_view);

            Kokkos::View<const T*, Kokkos::HostSpace> dbasis_view(dbasis, nq * nq);
            Kokkos::View<T*> d_dbasis("d_dbasis", nq * nq);
            Kokkos::deep_copy(d_dbasis, dbasis_view);

            Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nq*nq*nq * 6);
            Kokkos::View<T*> d_G("d_G", nelmt * nq * nq * nq * 6);
            Kokkos::deep_copy(d_G, G_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * ncomp * nm*nm*nm);
            Kokkos::View<T*> d_in("d_in", nelmt * ncomp * nm * nm * nm);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * ncomp * nm*nm*nm);
            Kokkos::View<T*> d_out("d_out", nelmt * ncomp * nm * nm * nm);

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<double>::max();

            //Kokkos with shared memory
            unsigned int ssize = nm * nq + nq * nq + 5 * nelmtPerBatch * nq * nq * nq;          
            
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

                    //shared memory access
                    T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);

                    T *s_basis  = scratch;
                    T *s_dbasis = s_basis  + nq * nm;

                    T *s_wsp0    = s_dbasis + nq * nq;
                    T *s_wsp1    = s_wsp0   + nelmtPerBatch * nq * nq * nq;

                    T *s_rqr     = s_wsp1 + nelmtPerBatch * nq * nq * nq;
                    T *s_rqs     = s_rqr  + nelmtPerBatch * nq * nq * nq;
                    T *s_rqt     = s_rqs  + nelmtPerBatch * nq * nq * nq;

                    const unsigned int threadIdx = team_member.team_rank();
                    const unsigned int blockSize = team_member.team_size();

                    //copy to shared memory
                    for(unsigned int tid = threadIdx; tid < nm * nq; tid += blockSize)
                    {
                        s_basis[tid] = d_basis[tid];
                    }
                
                
                    for(unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
                    {
                        s_dbasis[tid] = d_dbasis[tid];
                    }
                    team_member.team_barrier();
                        

                    /*
                    Interpolate to GL nodes
                    */

                    //element batch iteration
                    size_t eb = team_member.league_rank();
                    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
                    {   
                        //current nelmtPerBatch (edge case, last batch size can be less)
                        int remaining = nelmt - eb * nelmtPerBatch;
                        int c_nelmtPerBatch = (nelmtPerBatch < remaining) ? nelmtPerBatch : remaining;

                        //step-1 : Copy from in to s_rqr (comp 0), s_rqs (comp 1), s_rqt (comp 2)
                        for(int i = threadIdx; i < c_nelmtPerBatch * nm * nm * nm; i += blockSize)
                        {
                            int e = i / (nm * nm * nm);
                            int dof = i % (nm * nm * nm);

                            s_rqr[i] = d_in[eb * 3 * nelmtPerBatch * nm * nm * nm + e * 3 * nm * nm * nm + 0 * nm * nm * nm + dof];
                            s_rqs[i] = d_in[eb * 3 * nelmtPerBatch * nm * nm * nm + e * 3 * nm * nm * nm + 1 * nm * nm * nm + dof];
                            s_rqt[i] = d_in[eb * 3 * nelmtPerBatch * nm * nm * nm + e * 3 * nm * nm * nm + 2 * nm * nm * nm + dof];
                        }
                        team_member.team_barrier();
                        
                        //step-2 : component 0 - direction 0
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int i=0; i<nm; ++i){
                                r_p[i] = s_rqr[e * nm*nm*nm + i * nm*nm + j * nm + k];
                            }
                        
                            for (int p = 0; p < nq; ++p) {
                                T tmp = 0.0;
                            
                                for(int i = 0; i < nm; ++i) {
                                    tmp += s_basis[i * nq + p] * r_p[i];
                                }
                            
                                s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();
                            
                        //step-3 : component 0 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nm);
                            int p = (tid / nm) % nq;
                            int k = tid % nm;
                            for(int j=0; j<nm; ++j){
                                r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                            }
                        
                            for (int q = 0; q < nq; ++q) {
                                T tmp = 0.0;
                            
                                for(int j = 0; j < nm; ++j) {
                                    tmp += s_basis[j * nq + q] * r_q[j];
                                }
                            
                                s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();
                            
                        //step-4 : component 0 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int k=0; k<nm; ++k){
                                r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                            }
                            for (int r = 0; r < nq; ++r) {
                                T tmp = 0.0;
                            
                                for(int k = 0; k < nm; ++k) {
                                    tmp += s_basis[k * nq + r] * r_r[k];
                                }
                                
                                s_rqr[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();
                        
                        //step-5 : component 1 - direction 0
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int i=0; i<nm; ++i){
                                r_p[i] = s_rqs[e * nm*nm*nm + i * nm*nm + j * nm + k];
                            }
                        
                            for (int p = 0; p < nq; ++p) {
                                T tmp = 0.0;
                            
                                for(int i = 0; i < nm; ++i) {
                                    tmp += s_basis[i * nq + p] * r_p[i];
                                }
                            
                                s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //step-6 : component 1 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nm);
                            int p = (tid / nm) % nq;
                            int k = tid % nm;
                            for(int j=0; j<nm; ++j){
                                r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                            }
                        
                            for (int q = 0; q < nq; ++q) {
                                T tmp = 0.0;
                            
                                for(int j = 0; j < nm; ++j) {
                                    tmp += s_basis[j * nq + q] * r_q[j];
                                }
                            
                                s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //step-7 : component 1 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int k=0; k<nm; ++k){
                                r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                            }
                            for (int r = 0; r < nq; ++r) {
                                T tmp = 0.0;
                            
                                for(int k = 0; k < nm; ++k) {
                                    tmp += s_basis[k * nq + r] * r_r[k];
                                }
                                
                                s_rqs[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //step-8 : component 2 - direction 0
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int i=0; i<nm; ++i){
                                r_p[i] = s_rqt[e * nm*nm*nm + i * nm*nm + j * nm + k];
                            }
                        
                            for (int p = 0; p < nq; ++p) {
                                T tmp = 0.0;
                            
                                for(int i = 0; i < nm; ++i) {
                                    tmp += s_basis[i * nq + p] * r_p[i];
                                }
                            
                                s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //step-9 : component 2 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nm);
                            int p = (tid / nm) % nq;
                            int k = tid % nm;
                            for(int j=0; j<nm; ++j){
                                r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                            }
                        
                            for (int q = 0; q < nq; ++q) {
                                T tmp = 0.0;
                            
                                for(int j = 0; j < nm; ++j) {
                                    tmp += s_basis[j * nq + q] * r_q[j];
                                }
                            
                                s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //step-10 : component 2 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int k=0; k<nm; ++k){
                                r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                            }
                            for (int r = 0; r < nq; ++r) {
                                T tmp = 0.0;
                            
                                for(int k = 0; k < nm; ++k) {
                                    tmp += s_basis[k * nq + r] * r_r[k];
                                }
                                
                                s_rqt[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();


                        /*
                        Apply Quadrature Derivatives and Geometric Factors (Fused across 3 components)
                        */
                        T r_w0r[nq], r_w1r[nq], r_w1s[nq], r_w1t[nq], r_w2r[nq], r_w2s[nq], r_w2t[nq];

                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int r = tid % nq;

                            T r_p0[nq], r_p1[nq], r_p2[nq];

                            //copy to register
                            for(int n = 0; n < nq; n++)
                            {
                                r_p0[n] = s_rqr[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_p1[n] = s_rqs[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_p2[n] = s_rqt[e * nq*nq*nq + r * nq*nq + q * nq + n];
                                r_q[n]  = s_dbasis[n * nq + q];
                                r_r[n]  = s_dbasis[n * nq + r];
                            }
                        
                            T Grr, Grs, Grt, Gss, Gst, Gtt;
                        
                            for(int p = 0; p < nq; ++p){
                                T qr0 = 0, qs0 = 0, qt0 = 0;
                                T qr1 = 0, qs1 = 0, qt1 = 0;
                                T qr2 = 0, qs2 = 0, qt2 = 0;
                            
                                //Load Geometric Factors, coalesced access (loaded once for all 3 components)
                                Grr = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 0 * nq*nq*nq + p * nq * nq + q * nq + r];
                                Grs = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 1 * nq*nq*nq + p * nq * nq + q * nq + r];
                                Grt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 2 * nq*nq*nq + p * nq * nq + q * nq + r];
                                Gss = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 3 * nq*nq*nq + p * nq * nq + q * nq + r];
                                Gst = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 4 * nq*nq*nq + p * nq * nq + q * nq + r];
                                Gtt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 5 * nq*nq*nq + p * nq * nq + q * nq + r];
                            
                                // Multiply by D independently for each component
                                for(int n = 0; n < nq; n++){
                                    qr0 += s_dbasis[n * nq + p] * r_p0[n];
                                    qs0 += r_q[n] * s_rqr[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    qt0 += r_r[n] * s_rqr[e * nq*nq*nq + n * nq*nq + q * nq + p];

                                    qr1 += s_dbasis[n * nq + p] * r_p1[n];
                                    qs1 += r_q[n] * s_rqs[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    qt1 += r_r[n] * s_rqs[e * nq*nq*nq + n * nq*nq + q * nq + p];

                                    qr2 += s_dbasis[n * nq + p] * r_p2[n];
                                    qs2 += r_q[n] * s_rqt[e * nq*nq*nq + r * nq*nq + n * nq + p];
                                    qt2 += r_r[n] * s_rqt[e * nq*nq*nq + n * nq*nq + q * nq + p];
                                }
                            
                                // Apply chain rule for all 3 components
                                r_w0r[p] = Grr * qt0 + Grs * qs0 + Grt * qr0;
                                s_wsp0[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grs * qt0 + Gss * qs0 + Gst * qr0;
                                s_wsp1[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grt * qt0 + Gst * qs0 + Gtt * qr0;

                                r_w1r[p] = Grr * qt1 + Grs * qs1 + Grt * qr1;
                                r_w1s[p] = Grs * qt1 + Gss * qs1 + Gst * qr1;
                                r_w1t[p] = Grt * qt1 + Gst * qs1 + Gtt * qr1;

                                r_w2r[p] = Grr * qt2 + Grs * qs2 + Grt * qr2;
                                r_w2s[p] = Grs * qt2 + Gss * qs2 + Gst * qr2;
                                r_w2t[p] = Grt * qt2 + Gst * qs2 + Gtt * qr2;
                            }
                        }
                        team_member.team_barrier();

                        // Transpose derivative for Component 0 + stage Component 1 fluxes into s_rqs/s_rqt
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int r = tid % nq;
                            
                            for(int n = 0; n < nq; n++)
                            {
                                r_q[n] = s_dbasis[q * nq + n];
                                r_r[n] = s_dbasis[r * nq + n];
                            }
                        
                            for(int p = 0; p < nq; ++p)
                            {
                                T tmp0 = 0;
                                for(int n = 0; n < nq; ++n)
                                    tmp0 += r_w0r[n] * s_dbasis[p * nq + n];
                            
                                for(int n = 0; n < nq; ++n)                
                                    tmp0 += s_wsp0[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                            
                                for(int n = 0; n < nq; ++n)
                                    tmp0 += s_wsp1[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                            
                                s_rqr[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
                                s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = r_w1s[p];
                                s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = r_w1t[p];
                            }
                        }
                        team_member.team_barrier();

                        // Transpose derivative for Component 1 + stage Component 2 fluxes into s_wsp0/s_wsp1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int r = tid % nq;
                            
                            for(int n = 0; n < nq; n++)
                            {
                                r_q[n] = s_dbasis[q * nq + n];
                                r_r[n] = s_dbasis[r * nq + n];
                            }
                        
                            for(int p = 0; p < nq; ++p)
                            {
                                T tmp1 = 0;
                                for(int n = 0; n < nq; ++n)
                                    tmp1 += r_w1r[n] * s_dbasis[p * nq + n];
                            
                                for(int n = 0; n < nq; ++n)                
                                    tmp1 += s_rqs[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                            
                                for(int n = 0; n < nq; ++n)
                                    tmp1 += s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                            
                                r_p[p] = tmp1;
                                s_wsp0[e * nq*nq*nq + p * nq * nq + q * nq + r] = r_w2s[p];
                                s_wsp1[e * nq*nq*nq + p * nq * nq + q * nq + r] = r_w2t[p];
                            }
                        }
                        team_member.team_barrier();

                        // Write Component 1 output + Transpose derivative for Component 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int r = tid % nq;
                            
                            for(int n = 0; n < nq; n++)
                            {
                                r_q[n] = s_dbasis[q * nq + n];
                                r_r[n] = s_dbasis[r * nq + n];
                            }
                        
                            for(int p = 0; p < nq; ++p)
                            {
                                T tmp2 = 0;
                                for(int n = 0; n < nq; ++n)
                                    tmp2 += r_w2r[n] * s_dbasis[p * nq + n];
                            
                                for(int n = 0; n < nq; ++n)                
                                    tmp2 += s_wsp0[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                            
                                for(int n = 0; n < nq; ++n)
                                    tmp2 += s_wsp1[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                            
                                s_rqs[e * nq*nq*nq + r * nq*nq + q * nq + p] = r_p[p];
                                s_rqt[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp2;
                            }
                        }
                        team_member.team_barrier();


                        /*
                        Interpolate to GLL nodes
                        */
                        
                        //component 0 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {                
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int r=0; r<nq; ++r){
                                r_r[r] = s_rqr[e * nq*nq*nq + r * nq*nq + q * nq + p];
                            }
                            
                            for (int k = 0; k < nm; ++k) {
                                T tmp = 0.0;
                                
                                for(int r = 0; r < nq; ++r) {
                                    tmp += s_basis[k * nq + r] * r_r[r];
                                }
                                
                                s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                            }   
                        }
                        team_member.team_barrier();

                        //component 0 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {   
                            int e = tid / (nm * nq);
                            int k = (tid / nq) % nm;
                            int p = tid % nq;
                        
                            for(int q=0; q<nq; ++q){
                                r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                            }
                            
                            for (int j = 0; j < nm; ++j) {
                                T tmp = 0.0;
                            
                                for(int q = 0; q < nq; ++q) {
                                    tmp += s_basis[j * nq + q] * r_q[q];
                                }
                                s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //component 0 - direction 0 and write d_out
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {   
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int p=0; p<nq; ++p){
                                r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                            }
                        
                            for (int i = 0; i < nm; ++i) {
                                T tmp = 0.0;
                                for(int p = 0; p < nq; ++p) {
                                    tmp += s_basis[i * nq + p] * r_p[p];
                                }
                                d_out[eb * 3 * nelmtPerBatch * nm*nm*nm + e * 3*nm*nm*nm + 0 * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //component 1 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {                
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int r=0; r<nq; ++r){
                                r_r[r] = s_rqs[e * nq*nq*nq + r * nq*nq + q * nq + p];
                            }
                            
                            for (int k = 0; k < nm; ++k) {
                                T tmp = 0.0;
                                
                                for(int r = 0; r < nq; ++r) {
                                    tmp += s_basis[k * nq + r] * r_r[r];
                                }
                                
                                s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                            }   
                        }
                        team_member.team_barrier();

                        //component 1 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {   
                            int e = tid / (nm * nq);
                            int k = (tid / nq) % nm;
                            int p = tid % nq;
                        
                            for(int q=0; q<nq; ++q){
                                r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                            }
                            
                            for (int j = 0; j < nm; ++j) {
                                T tmp = 0.0;
                            
                                for(int q = 0; q < nq; ++q) {
                                    tmp += s_basis[j * nq + q] * r_q[q];
                                }
                                s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //component 1 - direction 0 and write d_out
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {   
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int p=0; p<nq; ++p){
                                r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                            }
                        
                            for (int i = 0; i < nm; ++i) {
                                T tmp = 0.0;
                                for(int p = 0; p < nq; ++p) {
                                    tmp += s_basis[i * nq + p] * r_p[p];
                                }
                                d_out[eb * 3 * nelmtPerBatch * nm*nm*nm + e * 3*nm*nm*nm + 1 * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //component 2 - direction 2
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                        {                
                            int e = tid / (nq * nq);
                            int q = (tid / nq) % nq;
                            int p = tid % nq;
                            
                            for(int r=0; r<nq; ++r){
                                r_r[r] = s_rqt[e * nq*nq*nq + r * nq*nq + q * nq + p];
                            }
                            
                            for (int k = 0; k < nm; ++k) {
                                T tmp = 0.0;
                                
                                for(int r = 0; r < nq; ++r) {
                                    tmp += s_basis[k * nq + r] * r_r[r];
                                }
                                
                                s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                            }   
                        }
                        team_member.team_barrier();

                        //component 2 - direction 1
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                        {   
                            int e = tid / (nm * nq);
                            int k = (tid / nq) % nm;
                            int p = tid % nq;
                        
                            for(int q=0; q<nq; ++q){
                                r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                            }
                            
                            for (int j = 0; j < nm; ++j) {
                                T tmp = 0.0;
                            
                                for(int q = 0; q < nq; ++q) {
                                    tmp += s_basis[j * nq + q] * r_q[q];
                                }
                                s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        //component 2 - direction 0 and write d_out
                        for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                        {   
                            int e = tid / (nm * nm);
                            int j = (tid / nm) % nm;
                            int k = tid % nm;
                        
                            for(int p=0; p<nq; ++p){
                                r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                            }
                        
                            for (int i = 0; i < nm; ++i) {
                                T tmp = 0.0;
                                for(int p = 0; p < nq; ++p) {
                                    tmp += s_basis[i * nq + p] * r_p[p];
                                }
                                d_out[eb * 3 * nelmtPerBatch * nm*nm*nm + e * 3*nm*nm*nm + 2 * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                            }
                        }
                        team_member.team_barrier();

                        eb += team_member.league_size();
                    }
                });
            Kokkos::fence();

            kokkosTimer.stop();
            const double t_w = kokkosTimer.elapsedSeconds();
            time_kokkos     = std::min(time_kokkos, t_w);
        }

        Kokkos::parallel_reduce(nelmt * ncomp * nm * nm * nm,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            sum);               
        sum = sum;

        T gdofPerSeconds = 1.0e-9 * nelmt * ncomp * nm * nm * nm / time_kokkos;
        results[0] = gdofPerSeconds; 
        results[1] = sum;
        results[2] = time_kokkos;
    }

    return results;
}


} //namespace Parallel
} //namespace BK4

#endif //BK4_TEMPLATED_FUSED_KOKKOS_KERNELS_HPP