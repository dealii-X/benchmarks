#ifndef KOKKOS_MIXEDGRAD_OPERATOR_HPP
#define KOKKOS_MIXEDGRAD_OPERATOR_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <algorithm>

namespace Parallel {

template<typename T, const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nm_p>
std::vector<double> Kokkos_MixedGrad(
    const unsigned int nelmt,  const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock,
    const T *__restrict__ dbasis_n, const T *__restrict__ basis_t, const T *__restrict__ basis_p,
    const T *__restrict__ G_scalar, T *__restrict__ in_p, T * __restrict__ out_u, const unsigned int ntests)
{
    constexpr unsigned int ndof_u_1D = nm_n * nm_t * nm_t;
    constexpr unsigned int ndof_u_total = ndof_u_1D * 3;
    
    constexpr unsigned int ndof_p_total = nm_p * nm_p * nm_p;

    T sum = 0.0;
    std::vector<double> results(3);
    {
        Kokkos::View<const T*, Kokkos::HostSpace> dbasis_n_view(dbasis_n, nm_n * nq);
        Kokkos::View<T*> d_dbasis_n("d_dbasis_n", nm_n * nq);
        Kokkos::deep_copy(d_dbasis_n, dbasis_n_view);

        Kokkos::View<const T*, Kokkos::HostSpace> basis_t_view(basis_t, nm_t * nq);
        Kokkos::View<T*> d_basis_t("d_basis_t", nm_t * nq);
        Kokkos::deep_copy(d_basis_t, basis_t_view);

        Kokkos::View<const T*, Kokkos::HostSpace> basis_p_view(basis_p, nm_p * nq);
        Kokkos::View<T*> d_basis_p("d_basis_p", nm_p * nq);
        Kokkos::deep_copy(d_basis_p, basis_p_view);

        // G_scalar size is purely nq*nq*nq (No element offset)
        Kokkos::View<const T*, Kokkos::HostSpace> G_view(G_scalar, nq*nq*nq);
        Kokkos::View<T*> d_G_scalar("d_G_scalar", nq * nq * nq);
        Kokkos::deep_copy(d_G_scalar, G_view);

        Kokkos::View<const T*, Kokkos::HostSpace> in_view(in_p, nelmt * ndof_p_total);
        Kokkos::View<T*> d_in("d_in", nelmt * ndof_p_total);
        Kokkos::deep_copy(d_in, in_view);

        Kokkos::View<T*, Kokkos::HostSpace> out_view(out_u, nelmt * ndof_u_total);
        Kokkos::View<T*> d_out("d_out", nelmt * ndof_u_total);

        Timer kokkosTimer;
        double time_kokkos = std::numeric_limits<T>::max();

        // Kokkos with shared memory - Exactly 5 workspaces of size nq^3 per element
        unsigned int ssize = nm_n * nq +
                             nm_t * nq +
                             nm_p * nq +
                             5 * nelmtPerBatch * nq * nq * nq;

        const unsigned int shmem_size = ssize * sizeof(T);

        // Calculate max size needed for thread-local register array
        constexpr unsigned int max_p1 = (nq > nm_n) ? nq : nm_n;
        constexpr unsigned int max_p2 = (nm_t > nm_p) ? nm_t : nm_p;
        constexpr unsigned int max_rp = (max_p1 > max_p2) ? max_p1 : max_p2;

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            kokkosTimer.start();
            Kokkos::parallel_for(policy,
            KOKKOS_LAMBDA (member_type team_member){
            
            // Thread-local register blocking array
            T r_p[max_rp]; 

            // Shared memory access
            T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
            T *s_dbasis_n = scratch;
            T *s_basis_t  = s_dbasis_n + nm_n * nq;
            T *s_basis_p  = s_basis_t  + nm_t * nq;

            T *s_wsp0    = s_basis_p + nm_p * nq;
            T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;
            T *s_accum   = s_wsp1    + nelmtPerBatch * nq * nq * nq; 
            T *s_out_0   = s_accum   + nelmtPerBatch * nq * nq * nq;
            T *s_out_1   = s_out_0   + nelmtPerBatch * nq * nq * nq;
            
            T *s_out_2   = s_accum; 

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();

            for(unsigned int tid = threadIdx; tid < nm_t * nq; tid += blockSize)
                s_basis_t[tid] = d_basis_t[tid];
        
            for(unsigned int tid = threadIdx; tid < nm_n * nq; tid += blockSize)
                s_dbasis_n[tid] = d_dbasis_n[tid];

            for(unsigned int tid = threadIdx; tid < nm_p * nq; tid += blockSize)
                s_basis_p[tid] = d_basis_p[tid];
            
            team_member.team_barrier();

            // Element batch iteration
            int eb = team_member.league_rank();
            
            while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {   
                const int global_batch_offset_in  = eb * nelmtPerBatch * ndof_p_total;
                const int global_batch_offset_out = eb * nelmtPerBatch * ndof_u_total;

                int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

                for(int tid = threadIdx; tid < c_nelmtPerBatch * ndof_p_total; tid += blockSize) {
                    int e = tid / ndof_p_total;
                    int dof = tid % ndof_p_total;
                    s_wsp0[tid] = d_in[global_batch_offset_in + e * ndof_p_total + dof];
                }
                team_member.team_barrier();
                
                // ==========================================
                // PHASE 1: Interpolate Scalar Pressure to Nodes
                // (Uses basis_p for x, y, and z)
                // ==========================================

                // --- Interpolate x-direction ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_p * nm_p; tid += blockSize) {
                    int e = tid / (nm_p * nm_p);
                    int j = (tid / nm_p) % nm_p;
                    int k = tid % nm_p;
                    
                    for(int i = 0; i < nm_p; ++i){
                        r_p[i] = s_wsp0[e * ndof_p_total + i*nm_p*nm_p + j*nm_p + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_p; ++i) {
                            tmp += r_p[i] * s_basis_p[i*nq + p]; 
                        }
                        s_wsp1[e * (nq*nm_p*nm_p) + p*nm_p*nm_p + j*nm_p + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // --- Interpolate y-direction ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_p; tid += blockSize) {
                    int e = tid / (nq * nm_p);
                    int p = (tid / nm_p) % nq;
                    int k = tid % nm_p;
                    
                    for(int j = 0; j < nm_p; ++j) {
                        r_p[j] = s_wsp1[e * (nq*nm_p*nm_p) + p*nm_p*nm_p + j*nm_p + k];
                    }
                    for (int q = 0; q < nq; ++q) {
                        T tmp = 0.0;
                        for(int j = 0; j < nm_p; ++j)
                            tmp += r_p[j] * s_basis_p[j*nq + q];

                        s_wsp0[e * (nq*nq*nm_p) + q*nq*nm_p + p*nm_p + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // --- Interpolate z-direction ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int p = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int k = 0; k < nm_p; ++k) {
                        r_p[k] = s_wsp0[e * (nq*nq*nm_p) + q*nq*nm_p + p*nm_p + k];
                    }
                    for (int r = 0; r < nq; ++r) {
                        T tmp = 0.0;
                        for(int k = 0; k < nm_p; ++k)
                            tmp += r_p[k] * s_basis_p[k*nq + r];

                        s_accum[e * (nq*nq*nq) + p*nq*nq + q*nq + r] = tmp; 
                    }
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 2: Apply Scalar Metric
                // ==========================================
                for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq * nq; tid += blockSize){
                    s_accum[tid] *= d_G_scalar[tid % (nq * nq * nq)];
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 3: Project back to H(div) Velocity Space
                // ==========================================

                // ---------------------------------------------------------
                // --- COMPONENT 0 (Test with x-derivative) ---
                // ---------------------------------------------------------
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int p = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_accum[e * (nq*nq*nq) + p*nq*nq + q*nq + r]; 
                    }
                    for (int k = 0; k < nm_t; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r) {
                            tmp += r_p[r] * s_basis_t[k * nq + r];
                        }
                        s_wsp1[e * (nq*nq*nm_t) + k*nq*nq + p*nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int p = (tid / nm_t) % nq;
                    int k = tid % nm_t;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nq*nq*nm_t) + k*nq*nq + p*nq + q]; 
                    }
                    for (int j = 0; j < nm_t; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_t[j*nq + q];

                        s_wsp0[e * (nq*nm_t*nm_t) + j*nq*nm_t + p*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_t; tid += blockSize) {
                    int e = tid / (nm_t * nm_t);
                    int j = (tid / nm_t) % nm_t;
                    int k = tid % nm_t;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_wsp0[e * (nq*nm_t*nm_t) + j*nq*nm_t + p*nm_t + k];
                    }
                    for (int i = 0; i < nm_n; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p)
                            tmp += r_p[p] * s_dbasis_n[i*nq + p]; // DERIVATIVE

                        s_out_0[e * ndof_u_1D + i*nm_t*nm_t + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // ---------------------------------------------------------
                // --- COMPONENT 1 (Test with y-derivative) ---
                // ---------------------------------------------------------
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int p = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_accum[e * (nq*nq*nq) + p*nq*nq + q*nq + r]; 
                    }
                    for (int k = 0; k < nm_t; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r) {
                            tmp += r_p[r] * s_basis_t[k * nq + r];
                        }
                        s_wsp1[e * (nq*nq*nm_t) + k*nq*nq + p*nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_t; tid += blockSize) {
                    int e = tid / (nq * nm_t);
                    int p = (tid / nm_t) % nq;
                    int k = tid % nm_t;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nq*nq*nm_t) + k*nq*nq + p*nq + q]; 
                    }
                    for (int j = 0; j < nm_n; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_dbasis_n[j*nq + q];

                        s_wsp0[e * (nq*nm_n*nm_t) + j*nq*nm_t + p*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_n * nm_t; tid += blockSize) {
                    int e = tid / (nm_n * nm_t);
                    int j = (tid / nm_t) % nm_n;
                    int k = tid % nm_t;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_wsp0[e * (nq*nm_n*nm_t) + j*nq*nm_t + p*nm_t + k];
                    }
                    for (int i = 0; i < nm_t; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p)
                            tmp += r_p[p] * s_basis_t[i*nq + p]; 

                        s_out_1[e * ndof_u_1D + i*nm_n*nm_t + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // ---------------------------------------------------------
                // --- COMPONENT 2 (Test with z-derivative) ---
                // ---------------------------------------------------------
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int p = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_accum[e * (nq*nq*nq) + p*nq*nq + q*nq + r]; 
                    }
                    for (int k = 0; k < nm_n; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r) {
                            tmp += r_p[r] * s_dbasis_n[k * nq + r];
                        }
                        s_wsp1[e * (nq*nq*nm_n) + k*nq*nq + p*nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_n; tid += blockSize) {
                    int e = tid / (nq * nm_n);
                    int p = (tid / nm_n) % nq;
                    int k = tid % nm_n;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nq*nq*nm_n) + k*nq*nq + p*nq + q]; 
                    }
                    for (int j = 0; j < nm_t; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_t[j*nq + q];

                        s_wsp0[e * (nq*nm_t*nm_n) + j*nq*nm_n + p*nm_n + k] = tmp;
                    }
                }
                team_member.team_barrier();

                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_n; tid += blockSize) {
                    int e = tid / (nm_t * nm_n);
                    int j = (tid / nm_n) % nm_t;
                    int k = tid % nm_n;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_wsp0[e * (nq*nm_t*nm_n) + j*nq*nm_n + p*nm_n + k];
                    }
                    for (int i = 0; i < nm_t; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p)
                            tmp += r_p[p] * s_basis_t[i*nq + p]; 

                        // Safe Aliasing: s_out_2 overwrites s_accum
                        s_out_2[e * ndof_u_1D + i*nm_t*nm_n + j*nm_n + k] = tmp;
                    }
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 4: Write to Global Output Space
                // ==========================================
                for(int tid = threadIdx; tid < c_nelmtPerBatch * ndof_u_1D; tid += blockSize) {
                    int e = tid / ndof_u_1D;
                    int dof = tid % ndof_u_1D;

                    d_out[global_batch_offset_out + e * ndof_u_total + 0 * ndof_u_1D + dof] = s_out_0[tid];
                    d_out[global_batch_offset_out + e * ndof_u_total + 1 * ndof_u_1D + dof] = s_out_1[tid];
                    d_out[global_batch_offset_out + e * ndof_u_total + 2 * ndof_u_1D + dof] = s_out_2[tid];
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

            Kokkos::parallel_reduce(nelmt * ndof_u_total,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            sum);
            
            T gdofPerSeconds = 1.0e-9 * nelmt * ndof_p_total / time_kokkos;
            results[0] = gdofPerSeconds; 
            results[1] = sum;
            results[2] = time_kokkos;
        }

        return results;
}

}  //namespace Parallel
#endif //KOKKOS_MIXEDGRAD_OPERATOR_HPP