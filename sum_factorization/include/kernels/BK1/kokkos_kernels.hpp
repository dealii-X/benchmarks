#ifndef KOKKOS_KERNELS_HPP
#define KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>

namespace Parallel{
template <typename T>
std::vector<T> KokkosKernel(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const T *__restrict__ basis0, const T *__restrict__ basis1, const T *__restrict__ basis2, 
    const T* __restrict__ JxW, const T* __restrict__ in, T* __restrict__ out,
    const unsigned int numThreads, const unsigned int threadsPerBlock, const unsigned int nelmt, const unsigned int ntests)
    {   
        const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

        const unsigned int nm0 = nq0 - 1;
        const unsigned int nm1 = nq1 - 1;
        const unsigned int nm2 = nq2 - 1;
        
        T result_kokkos = 0.0;
        std::vector<T> results(2);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> basis0_view(basis0, nm0 * nq0);
            Kokkos::View<const T*, Kokkos::HostSpace> basis1_view(basis1, nm1 * nq1);
            Kokkos::View<const T*, Kokkos::HostSpace> basis2_view(basis2, nm2 * nq2);
            Kokkos::View<T*> d_basis0("d_basis0", nm0 * nq0);
            Kokkos::View<T*> d_basis1("d_basis1", nm1 * nq1);
            Kokkos::View<T*> d_basis2("d_basis2", nm2 * nq2);
            Kokkos::deep_copy(d_basis0, basis0_view);
            Kokkos::deep_copy(d_basis1, basis1_view);
            Kokkos::deep_copy(d_basis2, basis2_view);

            Kokkos::View<const T*, Kokkos::HostSpace> JxW_view(JxW, nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_JxW("d_JxW", nelmt * nq0 * nq1 * nq2);
            Kokkos::deep_copy(d_JxW, JxW_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * nm0 * nm1 * nm2);
            Kokkos::View<T*> d_in("d_in", nelmt * nm0 * nm1 * nm2);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * nm0 * nm1 * nm2);
            Kokkos::View<T*> d_out("d_out", nelmt * nm0 * nm1 * nm2);

            Kokkos::fence();

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            //Kokkos with shared memory
            const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;         
            
            const unsigned int shmem_size = ssize * sizeof(T);
            
            typedef Kokkos::TeamPolicy<>::member_type member_type;
            Kokkos::TeamPolicy<> policy(numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock));
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
            
            for (unsigned int t = 0u; t < ntests; ++t)
            {
                kokkosTimer.start();
                Kokkos::parallel_for(policy,
                    KOKKOS_LAMBDA (member_type team_member){
                        //element index
                        unsigned int e = team_member.league_rank();

                        //shared memory access
                        T* scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
                        T* s_basis0 = scratch;
                        T* s_basis1 = s_basis0 + nm0 * nq0;
                        T* s_basis2 = s_basis1 + nm1 * nq1;
                        T* s_wsp0 = s_basis2 + nm2 * nq2;
                        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();
                        for(unsigned int tid = threadIdx; tid < nm0 * nq0; tid += blockSize)
                            s_basis0[tid] = d_basis0(tid);
                        for(unsigned int tid = threadIdx; tid < nm1 * nq1; tid += blockSize)
                            s_basis1[tid] = d_basis1(tid);
                        for(unsigned int tid = threadIdx; tid < nm2 * nq2; tid += blockSize)
                            s_basis2[tid] = d_basis2(tid);
                        
                        int i, j, k, p, q, r;
                        while(e < nelmt)
                        {   
                            //step-1 : Copy from in to the wsp0             
                            for(unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
                            {
                                s_wsp0[tid] = d_in(e * nm0 * nm1 * nm2 + tid);
                            }
                            team_member.team_barrier();

                            //step-2 : direction 0
                            for(unsigned int tid = threadIdx; tid < nq0 * nm1 * nm2; tid += blockSize)
                            {
                                p = tid / (nm1 * nm2);
                                j = (tid % (nm1 * nm2)) / nm2;
                                k =  tid % nm2;

                                T tmp = 0.0;
                                for(unsigned int i = 0; i < nm0; ++i)
                                {
                                    tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                                }

                                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();

                            //step-3 : direction 1
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2; tid += blockSize)
                            {
                                q = tid / (nq0 * nm2);
                                p = (tid % (nq0 * nm2)) / nm2;
                                k = tid % nm2;

                                T tmp = 0.0;
                                for(unsigned int j = 0; j < nm1; j++)
                                {
                                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                                }
                                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();

                            //step-4 : direction 2
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2; tid += blockSize)
                            {
                                p = tid / (nq1 * nq2);
                                q = (tid % (nq1 * nq2)) / nq2;
                                r = tid % nq2;

                                T tmp = 0.0;
                                for(unsigned int k = 0; k < nm2; ++k)
                                {
                                    tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                                }
                                s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
                            }
                            team_member.team_barrier();
                            
                            //Reverse Operations
                            
                            //step-5 : Multiply with weights and determinant of Jacobi
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
                                s_wsp1[tid] *= d_JxW[e * nq0 * nq1 * nq2 + tid];
                            }
                            team_member.team_barrier();
                            
                            //step-6 : direction 2
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2; tid += blockSize)
                            {
                                q = tid / (nq0 * nm2);
                                p = (tid % (nq0 * nm2)) / nm2;
                                k = tid % nm2;
                            
                                T tmp = 0.0;
                                for(unsigned int r = 0; r < nq2; ++r)
                                {
                                    tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                                }
                                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();

                            //step-7 : direction 1
                            for(unsigned int tid = threadIdx; tid < nm1 * nm2 * nq0; tid += blockSize)
                            {
                                p = tid / (nm1 * nm2);
                                j = (tid % (nm1 * nm2)) / nm2;
                                k = tid % nm2;
                            
                                T tmp = 0.0;
                                for(unsigned int q = 0; q < nq1; q++)
                                {
                                    tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                                }
                                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();


                            //step-8 : direction 0
                            for(unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
                            {
                                i = tid / (nm1 * nm2);
                                j = (tid % (nm1 * nm2)) / nm2;
                                k = tid % nm2;
                            
                                T tmp = 0.0;
                                for(unsigned int p = 0; p < nq0; ++p)
                                {
                                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                                }
                                s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();


                            //step-9 : Copy wsp0 to out
                            for(unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
                            {
                                d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
                            }
                            team_member.team_barrier();
                        
                            e += team_member.league_size();
                        }
                });
                Kokkos::fence();
                kokkosTimer.stop();
                const double t_w = kokkosTimer.elapsedSeconds();
                time_kokkos     = std::min(time_kokkos, t_w);
            }

            Kokkos::parallel_reduce(nelmt * nm0 * nm1 * nm2,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                result_kokkos);               
            result_kokkos = std::sqrt(result_kokkos);

            T gdofPerSeconds = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time_kokkos;
            results[0] = gdofPerSeconds; results[1] = result_kokkos;
        }

        return results;
    }
} //namespace Parallel

#endif //KOKKOS_KERNELS_HPP