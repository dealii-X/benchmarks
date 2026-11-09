#ifndef BK2_TEMPLATED_KOKKOS_KERNELS_HPP
#define BK2_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>
#include <limits>
#include <algorithm>

namespace BK2{
namespace Parallel{

template <typename T, const unsigned int nq>
std::vector<double> Kokkos_MassOperator(size_t nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const T* basis, const T* JxW, const T* in, T* out,
    const unsigned int ntests)
    {   
        const unsigned int nm = nq - 1;
        const unsigned int ncomp = 3;
        
        T sum = 0.0;
        std::vector<double> results(3);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> basis_view(basis, nm * nq);
            Kokkos::View<T*> d_basis("d_basis", nm * nq);
            Kokkos::deep_copy(d_basis, basis_view);

            Kokkos::View<const T*, Kokkos::HostSpace> JxW_view(JxW, nelmt * nq*nq*nq);
            Kokkos::View<T*> d_JxW("d_JxW", nelmt * nq*nq*nq);
            Kokkos::deep_copy(d_JxW, JxW_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * ncomp * nm*nm*nm);
            Kokkos::View<T*> d_in("d_in", nelmt * ncomp * nm*nm*nm);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * ncomp * nm*nm*nm);
            Kokkos::View<T*> d_out("d_out", nelmt * ncomp * nm*nm*nm);


            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<double>::max();

            //Kokkos with shared memory
            const unsigned int ssize = 3 * nelmtPerBatch * nq*nq*nq + nm*nq;         
            
            const unsigned int shmem_size = ssize * sizeof(T);
            
            typedef Kokkos::TeamPolicy<>::member_type member_type;
            Kokkos::TeamPolicy<> policy(numBlocks, threadsPerBlock);
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
            
            for (unsigned int t = 0u; t < ntests; ++t)
            {
                kokkosTimer.start();
                Kokkos::parallel_for(policy,
                    KOKKOS_LAMBDA (member_type team_member){
                        
                        T reg[nq];

                        //shared memory access
                        T* scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
                        T* s_basis = scratch;
                        T* s_wsp0 = s_basis + nm*nq;
                        T* s_wsp1 = s_wsp0 + nelmtPerBatch * nq*nq*nq;
                        T* s_JxW  = s_wsp1 + nelmtPerBatch * nq*nq*nq;

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();

                        for(int tid = threadIdx; tid < nm*nq; tid += blockSize)
                            s_basis[tid] = d_basis(tid);
                        team_member.team_barrier();

                        //element index
                        size_t eb = team_member.league_rank();
                        
                        while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
                        {   
                            //current nelmtPerBatch (edge case, last batch size can be less)
                            int remaining = nelmt - eb * nelmtPerBatch;
                            int c_nelmtPerBatch = (nelmtPerBatch < remaining) ? nelmtPerBatch : remaining;

                            // copy of JxW from global to shared memory for this batch
                            for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq * nq; tid += blockSize)
                            {
                                s_JxW[tid] = d_JxW[eb * nelmtPerBatch * nq * nq * nq + tid];
                            }
                            team_member.team_barrier();

                            for(int c = 0; c < 3; ++c)
                            {
                                const size_t comp_offset  = c * nelmt * nm * nm * nm;
                                const size_t batch_offset = eb * nelmtPerBatch * nm * nm * nm;
                                       
                                //step-1 : Copy from in to the wsp0
                                for(int i = threadIdx; i < c_nelmtPerBatch * nm * nm * nm; i += blockSize)
                                {
                                    s_wsp0[i] = d_in[comp_offset + batch_offset + i];
                                }
                                team_member.team_barrier();

                                //step-2 : direction 0
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                                {
                                    int e = tid / (nm * nm);
                                    int j = tid % (nm * nm) / nm;
                                    int k = tid % nm;
                                
                                    for(int i=0; i<nm; ++i){
                                        reg[i] = s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k];
                                    }
                                
                                    for (int p = 0; p < nq; ++p) {
                                        T tmp = 0.0;
                                    
                                        for(int i = 0; i < nm; ++i) {
                                            tmp += s_basis[i * nq + p] * reg[i];
                                        }
                                    
                                        s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                                    }
                                }
                                team_member.team_barrier();

                                //step-3 : direction 1
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                                {
                                    int e = tid / (nq * nm);
                                    int p = tid % (nq * nm) / nm;
                                    int k = tid % nm;
                                
                                    for(int j=0; j<nm; ++j){
                                        reg[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
                                    }
                                
                                    for (int q = 0; q < nq; ++q) {
                                        T tmp = 0.0;
                                    
                                        for(int j = 0; j < nm; ++j) {
                                            tmp += s_basis[j * nq + q] * reg[j];
                                        }
                                    
                                        s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                                    }
                                }
                                team_member.team_barrier();

                                //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                                {
                                    int e = tid / (nq * nq);
                                    int q = tid % (nq * nq) / nq;
                                    int p = tid % nq;
                                
                                    for(int k=0; k<nm; ++k){
                                        reg[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
                                    }
                                    for (int r = 0; r < nq; ++r) {
                                        T tmp = 0.0;
                                    
                                        for(int k = 0; k < nm; ++k) {
                                            tmp += s_basis[k * nq + r] * reg[k];
                                        }
                                    
                                        s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp * s_JxW[e * nq*nq*nq + r * nq*nq + q * nq + p];
                                    }
                                }
                                team_member.team_barrier();
                                
                                //Reverse Operations               
                                
                                //step-6 : direction 2
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize)
                                {                
                                    int e = tid / (nq * nq);
                                    int q = tid % (nq * nq) / nq;
                                    int p = tid % nq;
                                
                                    for(int r=0; r<nq; ++r){
                                        reg[r] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p];
                                    }
                                
                                    for (int k = 0; k < nm; ++k) {
                                        T tmp = 0.0;
                                    
                                        for(int r = 0; r < nq; ++r) {
                                            tmp += s_basis[k * nq + r] * reg[r];
                                        }
                                    
                                        s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                                    }   
                                }
                                team_member.team_barrier();

                                //step-7 : direction 1
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nq; tid += blockSize)
                                {   
                                    int e = tid / (nm * nq);
                                    int k = tid % (nm * nq) / nq;
                                    int p = tid % nq;
                                
                                    for(int q=0; q<nq; ++q){
                                        reg[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                                    }
                                
                                    for (int j = 0; j < nm; ++j) {
                                        T tmp = 0.0;
                                    
                                        for(int q = 0; q < nq; ++q) {
                                            tmp += s_basis[j * nq + q] * reg[q];
                                        }
                                        s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                                    }
                                }
                                team_member.team_barrier();


                                //step-8 : direction 0
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm; tid += blockSize)
                                {   
                                    int e = tid / (nm * nm);
                                    int j = tid % (nm * nm) / nm;
                                    int k = tid % nm;
                                
                                    for(int p=0; p<nq; ++p){
                                        reg[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                                    }
                                
                                    for (int i = 0; i < nm; ++i) {
                                        T tmp = 0.0;
                                        for(int p = 0; p < nq; ++p) {
                                            tmp += s_basis[i * nq + p] * reg[p];
                                        }
                                        s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                                    }
                                }
                                team_member.team_barrier();


                                //step-9 : Copy wsp0 to out
                                for(int tid = threadIdx; tid < c_nelmtPerBatch * nm * nm * nm; tid += blockSize)
                                {
                                    d_out[comp_offset + batch_offset + tid] = s_wsp0[tid];
                                }
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

            Kokkos::parallel_reduce(nelmt * ncomp * nm*nm*nm,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                sum);               

            T gdofPerSeconds = 1.0e-9 * nelmt * ncomp * nm*nm*nm / time_kokkos;
            results[0] = gdofPerSeconds;
            results[1] = sum;
            results[2] = time_kokkos;
        }

        return results;
    }


} //namespace Parallel
} //namespace BK2

#endif //BK2_TEMPLATED_KOKKOS_KERNELS_HPP