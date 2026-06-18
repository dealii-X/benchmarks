#ifndef KOKKOS_MIXEDDIV_OPERATOR_HPP
#define KOKKOS_MIXEDDIV_OPERATOR_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <algorithm>

namespace Parallel {

template<typename T, const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nm_p>
std::vector<double> Kokkos_MixedDiv(
    const unsigned int nelmt,  const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock,
    const T *__restrict__ dbasis_n, const T *__restrict__ basis_t, const T *__restrict__ basis_p,
    const T *__restrict__ G_scalar, T *__restrict__ in_u, T * __restrict__ out_p, const unsigned int ntests)
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

        Kokkos::View<const T*, Kokkos::HostSpace> G_view(G_scalar, nq*nq*nq);
        Kokkos::View<T*> d_G_scalar("d_G_scalar", nq * nq * nq);
        Kokkos::deep_copy(d_G_scalar, G_view);

        Kokkos::View<const T*, Kokkos::HostSpace> in_view(in_u, nelmt * ndof_u_total);
        Kokkos::View<T*> d_in("d_in", nelmt * ndof_u_total);
        Kokkos::deep_copy(d_in, in_view);

        Kokkos::View<T*, Kokkos::HostSpace> out_view(out_p, nelmt * ndof_p_total);
        Kokkos::View<T*> d_out("d_out", nelmt * ndof_p_total);

        Timer kokkosTimer;
        double time_kokkos = std::numeric_limits<T>::max();

        //Kokkos with shared memory
        unsigned int ssize = nm_n * nq +
                             nm_t * nq +
                             nm_p * nq +
                             3 * nelmtPerBatch * nq * nq * nq;

        const unsigned int shmem_size = ssize * sizeof(T);

        typedef Kokkos::TeamPolicy<>::member_type member_type;
        Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
        policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            kokkosTimer.start();
            Kokkos::parallel_for(policy,
            KOKKOS_LAMBDA (member_type team_member){
            
            // Thread-local register blocking array
            T r_p[nq]; 

            //shared memory access
            T *scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
            T *s_dbasis_n = scratch;
            T *s_basis_t  = s_dbasis_n + nm_n * nq;
            T *s_basis_p  = s_basis_t  + nm_t * nq;

            T *s_wsp0    = s_basis_p + nm_p * nq;
            T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;

            T *s_accum     = s_wsp1   + nelmtPerBatch * nq * nq * nq; 

            const unsigned int threadIdx = team_member.team_rank();
            const unsigned int blockSize = team_member.team_size();

            //copy bases to shared memory
            for(unsigned int tid = threadIdx; tid < nm_t * nq; tid += blockSize)
                s_basis_t[tid] = d_basis_t[tid];
        
            for(unsigned int tid = threadIdx; tid < nm_n * nq; tid += blockSize)
                s_dbasis_n[tid] = d_dbasis_n[tid];

            for(unsigned int tid = threadIdx; tid < nm_p * nq; tid += blockSize)
                s_basis_p[tid] = d_basis_p[tid];
            
            team_member.team_barrier();

            //element batch iteration
            int eb = team_member.league_rank();
            
            while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
            {   
                const int global_batch_offset = eb * nelmtPerBatch * ndof_u_total;
                int c_nelmtPerBatch = std::min(nelmtPerBatch, nelmt - eb * nelmtPerBatch);

 
                // ==========================================
                // PHASE 1: Compute Divergence 
                // ==========================================

                // --- Component 0 (x-direction) ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_t; tid += blockSize) {
                    int e = tid / (nm_t * nm_t);
                    int j = (tid / nm_t) % nm_t;
                    int k = tid % nm_t;
                    
                    for(int i = 0; i < nm_n; ++i){
                        r_p[i] = d_in[global_batch_offset + e * ndof_u_total + 0 * ndof_u_1D + i *nm_t*nm_t + j*nm_t + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_n; ++i) {
                            tmp += r_p[i] * s_dbasis_n[i*nq + p];
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

                        s_accum[e * (nq*nq*nq) + r*nq*nq + q*nq + p] = tmp;
                    }
                }
                team_member.team_barrier();
                
                // --- COMPONENT 1 (y-direction) ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_n * nm_t; tid += blockSize) {
                    int e = tid / (nm_n * nm_t);
                    int j = (tid / nm_t) % nm_n;
                    int k = tid % nm_t;
                    
                    for(int i = 0; i < nm_t; ++i) {
                        r_p[i] = d_in[global_batch_offset + e * ndof_u_total + 1 * ndof_u_1D + i *nm_t*nm_n + j*nm_t + k];
                    }
                    for (int p = 0; p < nq; ++p) {
                        T tmp = 0.0;
                        for(int i = 0; i < nm_t; ++i)
                            tmp += r_p[i] * s_basis_t[i*nq + p];

                        s_wsp1[e * (nq*nm_n*nm_t) + p*nm_n*nm_t + j*nm_t + k] = tmp;
                    }
                }
                team_member.team_barrier();

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
                            tmp += r_p[j] * s_dbasis_n[j*nq + q];

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

                        s_accum[e * (nq*nq*nq) + r*nq*nq + q*nq + p] += tmp;
                    }
                }
                team_member.team_barrier();

                // --- COMPONENT 2 (z-direction) ---
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_t * nm_n; tid += blockSize) {
                    int e = tid / (nm_t * nm_n);
                    int j = (tid / nm_n) % nm_t;
                    int k = tid % nm_n;
                    
                    for(int i = 0; i < nm_t; ++i) {
                        r_p[i] = d_in[global_batch_offset + e * ndof_u_total + 2 * ndof_u_1D + i *nm_t*nm_n + j*nm_n + k];
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
                            tmp += r_p[k] * s_dbasis_n[k*nq + r];

                        s_accum[e * (nq*nq*nq) + r*nq*nq + q*nq + p] += tmp;
                    }
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 2: Apply Scalar Geometric Metric
                // ==========================================
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq * nq; tid += blockSize){
                    s_accum[tid] *= d_G_scalar[tid % (nq * nq * nq)];
                }
                team_member.team_barrier();

                // ==========================================
                // PHASE 3: Project back to Scalar Pressure Space
                // ==========================================

                // Contraction 1 (x-direction)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize) {
                    int e = tid / (nq * nq);
                    int r = (tid / nq) % nq;
                    int q = tid % nq;
                    
                    for(int p = 0; p < nq; ++p) {
                        r_p[p] = s_accum[e * nq*nq*nq + r * nq*nq + q * nq + p]; 
                    }

                    for (int i = 0; i < nm_p; ++i) {
                        T tmp = 0.0;
                        for(int p = 0; p < nq; ++p) {
                            tmp += r_p[p] * s_basis_p[i * nq + p];
                        }
                        s_wsp0[e * (nq * nq * nm_p) + i * nq * nq + r * nq + q] = tmp;
                    }
                }
                team_member.team_barrier();

                // Contraction 2 (z-direction)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nm_p; tid += blockSize) {
                    int e = tid / (nq * nm_p);
                    int q = (tid / nm_p) % nq;
                    int i = tid % nm_p;
                    
                    for(int r = 0; r < nq; ++r) {
                        r_p[r] = s_wsp0[e * (nq * nq * nm_p) + i * nq * nq + r * nq + q]; 
                    }
                    
                    for (int k = 0; k < nm_p; ++k) {
                        T tmp = 0.0;
                        for(int r = 0; r < nq; ++r)
                            tmp += r_p[r] * s_basis_p[k*nq + r];

                        s_wsp1[e * (nm_p * nq * nm_p) + k * (nq * nm_p) + q * nm_p + i] = tmp;
                    }
                }
                team_member.team_barrier();

                // Contraction 3 (y-direction)
                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm_p * nm_p; tid += blockSize) {
                    int e = tid / (nm_p * nm_p);
                    int i = (tid / nm_p) % nm_p;
                    int k = tid % nm_p;
                    
                    for(int q = 0; q < nq; ++q) {
                        r_p[q] = s_wsp1[e * (nm_p * nq * nm_p) + k * (nq * nm_p) + q * nm_p + i];
                    }
                    
                    for (int j = 0; j < nm_p; ++j) {
                        T tmp = 0.0;
                        for(int q = 0; q < nq; ++q)
                            tmp += r_p[q] * s_basis_p[j*nq + q];

                        d_out[eb * nelmtPerBatch * ndof_p_total + e * ndof_p_total + i * nm_t*nm_t + j * nm_t + k] = tmp;
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

            Kokkos::parallel_reduce(nelmt * ndof_p_total,
            KOKKOS_LAMBDA(unsigned int i, T &val) {
                val += d_out(i) * d_out(i);
            },
            sum);
            
            T gdofPerSeconds = 1.0e-9 * nelmt * ndof_u_total / time_kokkos;
            results[0] = gdofPerSeconds; 
            results[1] = sum;
            results[2] = time_kokkos;
        }

        return results;
}

}  //namespace Parallel
#endif //KOKKOS_MIXEDDIV_OPERATOR_HPP