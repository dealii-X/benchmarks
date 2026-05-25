#ifndef KOKKOS_MASSOPERATOR_HPP
#define KOKKOS_MASSOPERATOR_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>

namespace Parallel {

template<typename T, const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n>
std::vector<double> Kokkos_Mass(
    const unsigned int nelmt,  const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock,
    const T *__restrict__ basis_n, const T *__restrict__ basis_t,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out, const unsigned int ntests)
{
    // Total DoFs per element for a Raviart-Thomas vector
    constexpr unsigned int ndof_1D = nm_n * nm_t * nm_t;
    constexpr unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

    T sum = 0.0;
    std::vector<double> results(3);
    {
        Kokkos::View<const T*, Kokkos::HostSpace> basis_n_view(basis_n, nm_n * nq);
        Kokkos::View<T*> d_basis_n("d_basis_n", nm_n * nq);
        Kokkos::deep_copy(d_basis_n, basis_n_view);

        Kokkos::View<const T*, Kokkos::HostSpace> basis_t_view(basis_t, nm_t * nq);
        Kokkos::View<T*> d_basis_t("d_basis_t", nm_t * nq);
        Kokkos::deep_copy(d_basis_t, basis_t_view);

        Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nq*nq*nq * 6);
        Kokkos::View<T*> d_G("d_G", nelmt * nq * nq * nq * 6);

        Kokkos::deep_copy(d_G, G_view);
        Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * ndof_total);
        Kokkos::View<T*> d_in("d_in", nelmt * ndof_total);

        Kokkos::deep_copy(d_in, in_view);
        Kokkos::View<T*, Kokkos::HostSpace> out_view(out, nelmt * ndof_total);
        Kokkos::View<T*> d_out("d_out", nelmt * ndof_total);


        Timer kokkosTimer;
        double time_kokkos = std::numeric_limits<T>::max();

        //Kokkos with shared memory
        unsigned int ssize = nm_n * nq +
                             nm_t * nq +
                             5 * nelmtPerBatch * nq * nq * nq;

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

            //shared memory access
            T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
            T *s_basis_n  = scratch;
            T *s_basis_t  = s_basis_n + nm_n * nq;

            T *s_wsp0    = s_basis_t + nm_t * nq;
            T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;

            T *s_uq_0     = s_wsp1   + nelmtPerBatch * nq * nq * nq;
            T *s_uq_1     = s_uq_0   + nelmtPerBatch * nq * nq * nq;
            T *s_uq_2     = s_uq_1   + nelmtPerBatch * nq * nq * nq;;

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();


            //copy to shared memory
            for(unsigned int tid = threadIdx; tid < nm_t * nq; tid += blockSize)
            {
                s_basis_t[tid] = d_basis_t[tid];
            }
        
        
            for(unsigned int tid = threadIdx; tid < nm_n * nq; tid += blockSize)
            {
                s_basis_n[tid] = d_basis_n[tid];
            }
            team_member.team_barrier();


            
            //element batch iteration
            int eb = team_member.league_rank();
            
            while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {   
                const int global_batch_offset = eb * nelmtPerBatch * ndof_total;

                //current nelmtPerBatch (edge case, last batch size can be less)
                int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);


                for(int tid = threadIdx; tid < c_nelmtPerBatch * ndof_1D; tid += blockSize) {
                    int e = tid / ndof_1D;
                    int dof = tid % ndof_1D;

                    s_uq_0[tid] = d_in[global_batch_offset + e * ndof_total + 0 * ndof_1D + dof];
                    s_uq_1[tid] = d_in[global_batch_offset + e * ndof_total + 1 * ndof_1D + dof];
                    s_uq_2[tid] = d_in[global_batch_offset + e * ndof_total + 2 * ndof_1D + dof];
                }
                team_member.team_barrier();
                
                // ==========================================
                // PHASE 1: Interpolate to Quadrature Nodes
                // ==========================================


                // --- Component 0 (x-direction) ---
                // x is normal (basis_n), y and z are tangent (basis_t)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_t; tid += blockSize) {
                    int e = tid / (nm_t * nm_t);
                    int j = (tid / nm_t) % nm_t;
                    int k = tid % nm_t;
                    
                    for(int i = 0; i < nm_n; ++i){
                        r_p[i] = s_uq_0[e * ndof_1D + i*nm_t*nm_t + j*nm_t + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_n; ++i) {
                            tmp += r_p[i] * s_basis_n[i*nq + p];
                        }
                        s_wsp1[e * (nq*nm_t*nm_t) + p*nm_t*nm_t + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int p = (tid / nm_t) % nq;
                    int k = tid % nm_t;
                    
                    for(int j = 0; j < nm_t; ++j) {
                        r_p[j] = s_wsp1[e * (nq*nm_t*nm_t) + p*nm_t*nm_t + j*nm_t + k];
                    }
                    for (int q = 0; q < nq; ++q) {
                        T tmp = 0.0;
                        for(int j = 0; j < nm_t; ++j)
                            tmp += r_p[j] * s_basis_t[j*nq + q];

                        s_wsp0[e * (nq*nq*nm_t) + q*nq*nm_t + p*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int q = (tid / nq) % nq;
                    int p = tid % nq;
                    
                    for(int k = 0; k < nm_t; ++k) {
                        r_p[k] = s_wsp0[e * (nq*nq*nm_t) + q*nq*nm_t + p*nm_t + k];
                    }
                    for (int r = 0; r < nq; ++r) {
                        T tmp = 0.0;
                        for(int k = 0; k < nm_t; ++k)
                            tmp += r_p[k] * s_basis_t[k*nq + r];

                        s_uq_0[e * (nq*nq*nq) + r*nq*nq + q*nq + p] = tmp;
                    }
                }
                team_member.team_barrier();
                

                // ---------------------------------------------------------
                // --- COMPONENT 1 (y-direction) ---
                // y is normal (basis_n), x and z are tangent (basis_t)
                // ---------------------------------------------------------
                
                // Contraction 1 (x-direction, uses basis_t)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_n * nm_t; tid += blockSize) {
                    int e = tid / (nm_n * nm_t);
                    int j = (tid / nm_t) % nm_n;
                    int k = tid % nm_t;
                    
                    for(int i = 0; i < nm_t; ++i) {
                        r_p[i] = s_uq_1[e * ndof_1D + i*nm_n*nm_t + j*nm_t + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_t; ++i)
                            tmp += r_p[i] * s_basis_t[i*nq + p];

                        s_wsp1[e * (nq*nm_n*nm_t) + p*nm_n*nm_t + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // Contraction 2 (y-direction, uses basis_n)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int p = (tid / nm_t) % nq;
                    int k = tid % nm_t;
                    
                    for(int j = 0; j < nm_n; ++j) {
                        r_p[j] = s_wsp1[e * (nq*nm_n*nm_t) + p*nm_n*nm_t + j*nm_t + k];
                    }
                    for (int q = 0; q < nq; ++q) {
                        T tmp = 0.0;
                        for(int j = 0; j < nm_n; ++j)
                            tmp += r_p[j] * s_basis_n[j*nq + q];

                        s_wsp0[e * (nq*nq*nm_t) + q*nq*nm_t + p*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // Contraction 3 (z-direction, uses basis_t)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int q = (tid / nq) % nq;
                    int p = tid % nq;
                    
                    for(int k = 0; k < nm_t; ++k) {
                        r_p[k] = s_wsp0[e * (nq*nq*nm_t) + q*nq*nm_t + p*nm_t + k];
                    }
                    for (int r = 0; r < nq; ++r) {
                        T tmp = 0.0;
                        for(int k = 0; k < nm_t; ++k)
                            tmp += r_p[k] * s_basis_t[k*nq + r];

                        s_uq_1[e * (nq*nq*nq) + r*nq*nq + q*nq + p] = tmp;
                    }
                }
                team_member.team_barrier();


                // ---------------------------------------------------------
                // --- COMPONENT 2 (z-direction) ---
                // z is normal (basis_n), x and y are tangent (basis_t)
                // ---------------------------------------------------------

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_n; tid += blockSize) {
                    int e = tid / (nm_t * nm_n);
                    int j = (tid / nm_n) % nm_t;
                    int k = tid % nm_n;
                    
                    for(int i = 0; i < nm_t; ++i) {
                        r_p[i] = s_uq_2[e * ndof_1D + i*nm_t*nm_n + j*nm_n + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_t; ++i)
                            tmp += r_p[i] * s_basis_t[i*nq + p];

                        s_wsp1[e * (nq*nm_t*nm_n) + p*nm_t*nm_n + j*nm_n + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_n; tid += blockSize) {
                    int e = tid / (nq * nm_n);
                    int p = (tid / nm_n) % nq;
                    int k = tid % nm_n;
                    
                    for(int j = 0; j < nm_t; ++j) {
                        r_p[j] = s_wsp1[e * (nq*nm_t*nm_n) + p*nm_t*nm_n + j*nm_n + k];
                    }
                    for (int q = 0; q < nq; ++q) {
                        T tmp = 0.0;
                        for(int j = 0; j < nm_t; ++j)
                            tmp += r_p[j] * s_basis_t[j*nq + q];

                        s_wsp0[e * (nq*nq*nm_n) + q*nq*nm_n + p*nm_n + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int q = (tid / nq) % nq;
                    int p = tid % nq;
                    
                    for(int k = 0; k < nm_n; ++k) {
                        r_p[k] = s_wsp0[e * (nq*nq*nm_n) + q*nq*nm_n + p*nm_n + k];
                    }
                    for (int r = 0; r < nq; ++r) {
                        T tmp = 0.0;
                        for(int k = 0; k < nm_n; ++k)
                            tmp += r_p[k] * s_basis_n[k*nq + r];

                        s_uq_2[e * (nq*nq*nq) + r*nq*nq + q*nq + p] = tmp;
                    }
                }
                team_member.team_barrier();


                // ==========================================
                // PHASE 2: Apply Piola Geometry Metric
                // ==========================================

                for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){

                    int e = tid / (nq * nq);
                    int q = (tid / nq) % nq; 
                    int p = tid % nq;  

                    int e_offset = eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq;

                    T G00, G01, G02, G11, G12, G22;
                    T u0, u1, u2;

                    for(unsigned int r = 0; r < nq; ++r){

                        int G_idx = e_offset + r * nq*nq + q * nq + p;

                        G00 = d_G[0 * nq*nq*nq + G_idx];
                        G01 = d_G[1 * nq*nq*nq + G_idx];
                        G02 = d_G[2 * nq*nq*nq + G_idx];
                        G11 = d_G[3 * nq*nq*nq + G_idx];
                        G12 = d_G[4 * nq*nq*nq + G_idx];
                        G22 = d_G[5 * nq*nq*nq + G_idx];
                    
                        int shm_idx = e * nq*nq*nq + r * nq*nq + q * nq + p;
                    
                        u0 = s_uq_0[shm_idx];
                        u1 = s_uq_1[shm_idx];
                        u2 = s_uq_2[shm_idx];
                    
                        s_uq_0[shm_idx] = G00 * u0 + G01 * u1 + G02 * u2;
                        s_uq_1[shm_idx] = G01 * u0 + G11 * u1 + G12 * u2;
                        s_uq_2[shm_idx] = G02 * u0 + G12 * u1 + G22 * u2;
                    }
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 3: Project back to Nodes (q and r directions)
                // ==========================================

                // ---------------------------------------------------------
                // --- COMPONENT 0 (x-direction) ---
                // Initial state from Phase 2: s_uq_0 has shape [e][i][q][r]
                // 'i' is nm_n (x-normal), 'q' is nq (y-tangent), 'r' is nq (z-tangent).
                // ---------------------------------------------------------

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int r = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_uq_0[e * nq*nq*nq + r * nq*nq + q * nq + p]; 
                    }

                    for (int i = 0; i < nm_n; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p) {
                            tmp += r_p[p] * s_basis_n[i * nq + p];
                        }
                        s_wsp0[e * (nq * nq * nm_n) + i * nq * nq + r * nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_n; tid += blockSize) {
                    int e = tid / (nq * nm_n);
                    int q = (tid / nm_n) % nq;
                    int i = tid % nm_n;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_wsp0[e * (nq * nq * nm_n) + i * nq * nq + r * nq + q]; 
                    }
                    
                    for (int k = 0; k < nm_t; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r)
                            tmp += r_p[r] * s_basis_t[k*nq + r];

                        s_wsp1[e * (nm_t * nq * nm_n) + k * (nq * nm_n) + q * nm_n + i] = tmp;
                    }
                }
                team_member.team_barrier();


                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_n * nm_t; tid += blockSize) {
                    int e = tid / (nm_n * nm_t);
                    int i = (tid / nm_t) % nm_n;
                    int k = tid % nm_t;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nm_t * nq * nm_n) + k * (nq * nm_n) + q * nm_n + i];
                    }
                    
                    for (int j = 0; j < nm_t; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_t[j*nq + q];

                        s_uq_0[e * (nm_n*nm_t*nm_t) + i*(nm_t*nm_t) + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                
                //----------------------------------------------
                // --- COMPONENT 1 (y-direction) ---
                // p uses basis_t, q uses basis_n, r uses basis_t
                // Final layout: i*nm_n*nm_t + j*nm_t + k
                // ---------------------------------------------------------

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int r = (tid / nq) % nq;
                    int q = tid % nq;

                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_uq_1[e * nq*nq*nq + r * nq*nq + q * nq + p]; 
                    }
                    
                    for (int i = 0; i < nm_t; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p) {
                            tmp += r_p[p] * s_basis_t[i * nq + p];
                        }
                        s_wsp0[e * (nq * nq * nm_t) + i * nq * nq + r * nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int q = (tid / nm_t) % nq;
                    int i = tid % nm_t;

                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_wsp0[e * (nq * nq * nm_t) + i * nq * nq + r * nq + q]; 
                    }
                
                    for (int k = 0; k < nm_t; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r) {
                            tmp += r_p[r] * s_basis_t[k * nq + r];
                        }
                    
                        s_wsp1[e * (nm_t * nq * nm_t) + k * (nq * nm_t) + q * nm_t + i] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_t; tid += blockSize) {
                    int e = tid / (nm_t * nm_t);
                    int i = (tid / nm_t) % nm_t;
                    int k = tid % nm_t;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nm_t * nq * nm_t) + k * (nq * nm_t) + q * nm_t + i];
                    }
                    
                    for (int j = 0; j < nm_n; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_n[j*nq + q]; 

                        s_uq_1[e * (nm_t*nm_n*nm_t) + i*(nm_n*nm_t) + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();



                // ---------------------------------------------------------
                // --- COMPONENT 2 (z-direction) ---
                // p uses basis_t, q uses basis_t, r uses basis_n
                // Final layout: i*nm_t*nm_n + j*nm_n + k
                // ---------------------------------------------------------
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int r = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_uq_2[e * nq*nq*nq + r * nq*nq + q * nq + p]; 
                    }

                    for (int i = 0; i < nm_t; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p) {
                            tmp += r_p[p] * s_basis_t[i * nq + p];
                        }
                        s_wsp1[e * (nq * nq * nm_t) + i * nq * nq + r * nq + q] = tmp;
                    }
                }
                team_member.team_barrier();


                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int q = (tid / nm_t) % nq;
                    int i = tid % nm_t;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_wsp1[e * (nq * nq * nm_t) + i * nq * nq + r * nq + q]; 
                    }
                    
                    for (int k = 0; k < nm_n; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r)
                            tmp += r_p[r] * s_basis_n[k*nq + r];

                        s_wsp0[e * (nm_t*nm_n*nq) + k*(nm_t*nq) + q*nm_t + i] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_n; tid += blockSize) {
                    int e = tid / (nm_t * nm_n);
                    int i = (tid / nm_n) % nm_t;
                    int k = tid % nm_n;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp0[e * (nm_t*nm_n*nq) + k*(nm_t*nq) + q*nm_t + i];
                    }
                    
                    for (int j = 0; j < nm_t; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_t[j*nq + q];

                        s_uq_2[e * (nm_t*nm_t*nm_n) + i*(nm_t*nm_n) + j*nm_n + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 4: Write to Output
                // ==========================================

                const int global_batch_offset_out = eb * nelmtPerBatch * ndof_total;

                for(int tid = threadIdx; tid < c_nelmtPerBatch * ndof_1D; tid += blockSize) {
                    // Unpack 1D thread ID into local element and DoF indices
                    int e = tid / ndof_1D;
                    int dof = tid % ndof_1D;

                    // Write X, Y, and Z components from shared memory to their precise global locations
                    d_out[global_batch_offset_out + e * ndof_total + 0 * ndof_1D + dof] = s_uq_0[tid];
                    d_out[global_batch_offset_out + e * ndof_total + 1 * ndof_1D + dof] = s_uq_1[tid];
                    d_out[global_batch_offset_out + e * ndof_total + 2 * ndof_1D + dof] = s_uq_2[tid];
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

            Kokkos::parallel_reduce(nelmt * ndof_total,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            sum);
            
            T gdofPerSeconds = 1.0e-9 * nelmt * ndof_total / time_kokkos;
                results[0] = gdofPerSeconds; 
                results[1] = sum;
                results[2] = time_kokkos;
            }

        return results;
    }
        

}  //namespace Parallel
#endif //KOKKOS_MASSOPERATOR_HPP