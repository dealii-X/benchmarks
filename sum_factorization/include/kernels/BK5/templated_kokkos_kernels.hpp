#ifndef BK5_TEMPLATED_KOKKOS_KERNELS_HPP
#define BK5_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>

namespace Parallel{
template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
std::vector<T> KokkosKernel_3D_Block_SimpleMap(
    const T *__restrict__ dbasis0, const T *__restrict__ dbasis1, const T *__restrict__ dbasis2,
    const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out,
    const unsigned int numThreads3D, const unsigned int nelmt, const unsigned int ntests)
    {   
        const unsigned int numBlocks = numThreads3D / (nq0 * nq1 * nq2);

        T result_kokkos = 0.0;
        std::vector<T> results(3);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis0_view(dbasis0, nq0 * nq0);
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis1_view(dbasis1, nq1 * nq1);
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis2_view(dbasis2, nq2 * nq2);
            Kokkos::View<T*> d_dbasis0("d_dbasis0", nq0 * nq0);
            Kokkos::View<T*> d_dbasis1("d_dbasis1", nq1 * nq1);
            Kokkos::View<T*> d_dbasis2("d_dbasis2", nq2 * nq2);
            Kokkos::deep_copy(d_dbasis0, dbasis0_view);
            Kokkos::deep_copy(d_dbasis1, dbasis1_view);
            Kokkos::deep_copy(d_dbasis2, dbasis2_view);

            Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nq0 * nq1 * nq2 * 6);
            Kokkos::View<T*> d_G("d_G", nelmt * nq0 * nq1 * nq2 * 6);
            Kokkos::deep_copy(d_G, G_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_in("d_in", nelmt * nq0 * nq1 * nq2);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_out("d_out", nelmt * nq0 * nq1 * nq2);

            Kokkos::fence();   //deep copies in Kokkos are async

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            //Kokkos with shared memory
            const unsigned int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;         
            
            const unsigned int shmem_size = ssize * sizeof(T);
            
            typedef Kokkos::TeamPolicy<>::member_type member_type;
            Kokkos::TeamPolicy<> policy(numBlocks, nq0 * nq1 * nq2);
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
                        T* s_dbasis0 = scratch;
                        T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
                        T *s_dbasis2 = s_dbasis1 + nq1 * nq1;
                        T *rqr     = s_dbasis2 + nq2 * nq2;
                        T *rqs     = rqr + nq0 * nq1 * nq2;
                        T *rqt     = rqs + nq0 * nq1 * nq2;

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();

                        for(unsigned int tid = threadIdx; tid < nq0 * nq0; tid += blockSize)
                        {
                            s_dbasis0[tid] = d_dbasis0(tid);
                        }
                        for(unsigned int tid = threadIdx; tid < nq1 * nq1; tid += blockSize)
                        {
                            s_dbasis1[tid] = d_dbasis1(tid);
                        }
                        for(unsigned int tid = threadIdx; tid < nq2 * nq2; tid += blockSize)
                        {
                            s_dbasis2[tid] = d_dbasis2(tid);
                        }
                        
                        team_member.team_barrier();
                        
                        T Grr, Grs, Grt, Gss, Gst, Gtt;
                        T qr, qs, qt;

                        while(e < nelmt)
                        {   
                            const unsigned int tid = threadIdx;

                            const unsigned int i = tid / (nq1 * nq2);
                            const unsigned int j = (tid % (nq1 * nq2)) / nq2;
                            const unsigned int k = tid % nq2;

                            qr = 0; qs = 0; qt = 0;
    
                            //Load Geometric Factors, coalesced access
                            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
    
                            // Multiply by D
                            for(unsigned int n = 0; n < nq0; n++){
                                qr += s_dbasis0[i * nq0 + n] * d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
                            }
    
                            for(unsigned int n = 0; n < nq1; n++){
                                qs += s_dbasis1[j * nq1 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                            }
    
                            for(unsigned int n = 0; n < nq2; n++){
                                qt += s_dbasis2[k * nq2 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
                            }
        
                            // Apply chain rule
                            rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                            rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                            rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
        
                            team_member.team_barrier();

                            T tmp0 = 0;
                            for(unsigned int n = 0; n < nq0; ++n)
                                tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * s_dbasis0[n * nq0 + i];
    
                            for(unsigned int n = 0; n < nq1; ++n)                
                                tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * s_dbasis1[n * nq1 + j];
    
                            for(unsigned int n = 0; n < nq2; ++n)
                                tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * s_dbasis2[n * nq2 + k];
                            
                            d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;

                            e += team_member.league_size();                            
                        }

                    }
                );
                Kokkos::fence();
                kokkosTimer.stop();
                const double t_w = kokkosTimer.elapsedSeconds();
                time_kokkos     = std::min(time_kokkos, t_w);
            }

            Kokkos::parallel_reduce(nelmt * nq0 * nq1 * nq2,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                result_kokkos);               
            result_kokkos = std::sqrt(result_kokkos);

            T gdofPerSeconds = 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time_kokkos;
            results[0] = gdofPerSeconds; 
            results[1] = result_kokkos;
            results[2] = time_kokkos;

        }

        return results;
    }



template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
std::vector<T> KokkosKernel_2D_Block_jk_SimpleMap(
    const T *__restrict__ dbasis0, const T *__restrict__ dbasis1, const T *__restrict__ dbasis2,
    const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out,
    const unsigned int numThreads3D, const unsigned int nelmt, const unsigned int ntests)
    {   
        const unsigned int numBlocks = (numThreads3D / nq0) / (nq1 * nq2);

        T result_kokkos = 0.0;
        std::vector<T> results(3);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis0_view(dbasis0, nq0 * nq0);
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis1_view(dbasis1, nq1 * nq1);
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis2_view(dbasis2, nq2 * nq2);
            Kokkos::View<T*> d_dbasis0("d_dbasis0", nq0 * nq0);
            Kokkos::View<T*> d_dbasis1("d_dbasis1", nq1 * nq1);
            Kokkos::View<T*> d_dbasis2("d_dbasis2", nq2 * nq2);
            Kokkos::deep_copy(d_dbasis0, dbasis0_view);
            Kokkos::deep_copy(d_dbasis1, dbasis1_view);
            Kokkos::deep_copy(d_dbasis2, dbasis2_view);

            Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nq0 * nq1 * nq2 * 6);
            Kokkos::View<T*> d_G("d_G", nelmt * nq0 * nq1 * nq2 * 6);
            Kokkos::deep_copy(d_G, G_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_in("d_in", nelmt * nq0 * nq1 * nq2);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_out("d_out", nelmt * nq0 * nq1 * nq2);

            Kokkos::fence();   //deep copies in Kokkos are async

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            //Kokkos with shared memory
            const unsigned int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;         
            
            const unsigned int shmem_size = ssize * sizeof(T);
            
            typedef Kokkos::TeamPolicy<>::member_type member_type;
            Kokkos::TeamPolicy<> policy(numBlocks, nq1 * nq2);
            policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));
            
            for (unsigned int t = 0u; t < ntests; ++t)
            {
                kokkosTimer.start();
                Kokkos::parallel_for(policy,
                    KOKKOS_LAMBDA (member_type team_member){

                        T r_i[nq0];
                        T r_j[nq1];
                        T r_k[nq2];
                        //element index
                        unsigned int e = team_member.league_rank();

                        //shared memory access
                        T* scratch = (T*)team_member.team_shmem().get_shmem(shmem_size);
                        T* s_dbasis0 = scratch;
                        T *rqr     = s_dbasis0 + nq0 * nq0;
                        T *rqs     = rqr + nq0 * nq1 * nq2;
                        T *rqt     = rqs + nq0 * nq1 * nq2;

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();

                        for(unsigned int tid = threadIdx; tid < nq0 * nq0; tid += blockSize)
                        {
                            s_dbasis0[tid] = d_dbasis0(tid);
                        }

                        team_member.team_barrier();

                        while(e < nelmt)
                        {   
                            const unsigned int tid = threadIdx;
        
                            unsigned int j = tid / nq2;
                            unsigned int k = tid % nq2;
                            
                            //copy to register
                            for(unsigned int n = 0; n < nq0; n++)
                            {
                                r_i[n] = d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
                            }
                        
                            for(unsigned int n = 0; n < nq1; n++)
                            {
                                r_j[n] = d_dbasis1[j * nq1 + n];
                            }
                        
                            for(unsigned int n = 0; n < nq2; n++)
                            {
                                r_k[n] = d_dbasis2[k * nq2 + n];
                            }
                            
                            T Grr, Grs, Grt, Gss, Gst, Gtt;
                            T qr, qs, qt;

                            for(unsigned int i = 0; i < nq0; ++i){

                                qr = 0; qs = 0; qt = 0; 

                                //Load Geometric Factors, coalesced access
                                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];

                                // Multiply by D
                                for(unsigned int n = 0; n < nq0; n++){
                                    qr += s_dbasis0[i * nq0 + n] * r_i[n];
                                }
                            
                                for(unsigned int n = 0; n < nq1; n++){
                                    qs += r_j[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                                }

                                for(unsigned int n = 0; n < nq2; n++){
                                    qt += r_k[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
                                }

                                // Apply chain rule
                                rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                                rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                                rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
                            }                   
                            team_member.team_barrier();
                            
                            //copy to register
                            for(unsigned int n = 0; n < nq0; n++)
                            {
                                r_i[n] = rqr[n * nq1 * nq2 + j * nq2 + k];
                            }
                        
                            for(unsigned int n = 0; n < nq1; n++)
                            {
                                r_j[n] = d_dbasis1[n * nq1 + j];
                            }
                        
                            for(unsigned int n = 0; n < nq2; n++)
                            {
                                r_k[n] = d_dbasis2[n * nq2 + k];
                            }

                           for(unsigned int i = 0; i < nq0; ++i){

                                T tmp0 = 0;
                                for(unsigned int n = 0; n < nq0; ++n)
                                    tmp0 += r_i[n] * s_dbasis0[n * nq0 + i];

                                for(unsigned int n = 0; n < nq1; ++n)                
                                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * r_j[n];

                                for(unsigned int n = 0; n < nq2; ++n)
                                    tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * r_k[n];

                                d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
                            }
                            e += team_member.league_size();                            
                        }

                    }
                );
                Kokkos::fence();
                kokkosTimer.stop();
                const double t_w = kokkosTimer.elapsedSeconds();
                time_kokkos     = std::min(time_kokkos, t_w);
            }

            Kokkos::parallel_reduce(nelmt * nq0 * nq1 * nq2,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                result_kokkos);               
            result_kokkos = std::sqrt(result_kokkos);

            T gdofPerSeconds = 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time_kokkos;
            results[0] = gdofPerSeconds;
            results[1] = result_kokkos;
            results[2] = time_kokkos;

        }

        return results;
    }

} //namespace Parallel

#endif //BK5_TEMPLATED_KOKKOS_KERNELS_HPP