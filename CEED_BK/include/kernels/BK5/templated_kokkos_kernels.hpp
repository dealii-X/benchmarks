#ifndef BK5_TEMPLATED_KOKKOS_KERNELS_HPP
#define BK5_TEMPLATED_KOKKOS_KERNELS_HPP

#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>

namespace BK5{
namespace Parallel{

template <typename T, const unsigned int nq>
std::vector<double> Kokkos_LaplaceOperator(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const T *__restrict__ dbasis,
    const T* __restrict__ G, const T* __restrict__ in, T* __restrict__ out, const unsigned int ntests)
    {   
        T sum = 0.0;
        std::vector<double> results(3);
        {   
            Kokkos::View<const T*, Kokkos::HostSpace> dbasis_view(dbasis, nq * nq);
            Kokkos::View<T*> d_dbasis("d_dbasis", nq * nq);
            Kokkos::deep_copy(d_dbasis, dbasis_view);


            Kokkos::View<const T*, Kokkos::HostSpace> G_view(G, nelmt * nq * nq * nq * 6);
            Kokkos::View<T*> d_G("d_G", nelmt * nq * nq * nq * 6);
            Kokkos::deep_copy(d_G, G_view);

            Kokkos::View<const T*, Kokkos::HostSpace> in_view(in, nelmt * nq * nq * nq);
            Kokkos::View<T*> d_in("d_in", nelmt * nq * nq * nq);
            Kokkos::deep_copy(d_in, in_view);

            Kokkos::View<const T*, Kokkos::HostSpace> out_view(out, nelmt * nq * nq * nq);
            Kokkos::View<T*> d_out("d_out", nelmt * nq * nq * nq);

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            //Kokkos with shared memory
            const unsigned int ssize = nq * nq + (4 * nelmtPerBatch * nq * nq * nq);         
            
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

                        //element index
                        unsigned int e = team_member.league_rank();

                        //shared memory access
                        T* scratch  = (T*)team_member.team_shmem().get_shmem(shmem_size);
                        T *s_dbasis = scratch;
                        T *s_rqr    = s_dbasis + nq * nq;
                        T *s_rqs    = s_rqr + nelmtPerBatch * nq * nq * nq;
                        T *s_rqt    = s_rqs + nelmtPerBatch * nq * nq * nq;
                        T *s_wsp    = s_rqt + nelmtPerBatch * nq * nq * nq;

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();
                        
                        //copy to shared memory
                        for(unsigned int tid = threadIdx; tid < nq * nq; tid += blockSize)
                        {
                            s_dbasis[tid] = d_dbasis(tid);
                        }
                        
                        team_member.team_barrier();
                        
                        T Grr, Grs, Grt, Gss, Gst, Gtt;
                        T qr, qs, qt;

                        int eb = team_member.league_rank();
                        while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
                        {   
                            //current nelmtPerBatch (edge case, last batch size can be less)
                            int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ? (nelmt - eb * nelmtPerBatch) : nelmtPerBatch; 
                        
                            //step-1 : Copy from in to the wsp0
                            for(int i = threadIdx; i < c_nelmtPerBatch * nq * nq * nq; i += blockSize)
                            {
                                s_wsp[i] = d_in[eb * nelmtPerBatch * nq * nq * nq + i];
                            }
                            team_member.team_barrier();
                            
                            
                            for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){

                                int b = tid / (nq * nq);
                                int q = tid % (nq * nq) / nq;
                                int r = tid % nq;

                                //copy to register
                                for(unsigned int n = 0; n < nq; n++)
                                {
                                    r_p[n] = s_wsp[b * nq*nq*nq + r * nq*nq + q * nq + n];
                                    r_q[n] = s_dbasis[q * nq + n];
                                    r_r[n] = s_dbasis[r * nq + n];
                                }
                            
                                T Grr, Grs, Grt, Gss, Gst, Gtt;
                                T qr, qs, qt;
                            
                                for(unsigned int p = 0; p < nq; ++p){
                                
                                    qr = 0; qs = 0; qt = 0; 
                                
                                    //Load Geometric Factors, coalesced access
                                    Grr = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 0 * nq*nq*nq + p * nq * nq + q * nq + r];
                                    Grs = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 1 * nq*nq*nq + p * nq * nq + q * nq + r];
                                    Grt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 2 * nq*nq*nq + p * nq * nq + q * nq + r];
                                    Gss = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 3 * nq*nq*nq + p * nq * nq + q * nq + r];
                                    Gst = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 4 * nq*nq*nq + p * nq * nq + q * nq + r];
                                    Gtt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 5 * nq*nq*nq + p * nq * nq + q * nq + r];
                                
                                    // Multiply by D
                                    for(unsigned int n = 0; n < nq; n++){
                                        qr += s_dbasis[p * nq + n] * r_p[n];
                                        qs += r_q[n] * s_wsp[b * nq*nq*nq + r * nq*nq + n * nq + p];
                                        qt += r_r[n] * s_wsp[b * nq*nq*nq + n * nq*nq + q * nq + p];
                                    }
                                
                                    // Apply chain rule
                                    s_rqr[b * nq*nq*nq + p * nq * nq + q * nq + r] = Grr * qr + Grs * qs + Grt * qt;
                                    s_rqs[b * nq*nq*nq + p * nq * nq + q * nq + r] = Grs * qr + Gss * qs + Gst * qt;
                                    s_rqt[b * nq*nq*nq + p * nq * nq + q * nq + r] = Grt * qr + Gst * qs + Gtt * qt;
                                }
                            }
                            team_member.team_barrier();

                    
                            for(unsigned int tid = threadIdx; tid < c_nelmtPerBatch * nq * nq; tid += blockSize){
                            
                                int b = tid / (nq * nq);
                                int q = tid % (nq * nq) / nq;
                                int r = tid % nq;
                            
                                //copy to register
                                for(unsigned int n = 0; n < nq; n++)
                                {
                                    r_p[n] = s_rqr[b * nq*nq*nq + n * nq * nq + q * nq + r];
                                    r_q[n] = s_dbasis[n * nq + q];
                                    r_r[n] = s_dbasis[n * nq + r];
                                }
                            
                                for(unsigned int p = 0; p < nq; ++p)
                                {
                                    T tmp0 = 0;
                                    for(unsigned int n = 0; n < nq; ++n)
                                        tmp0 += r_p[n] * s_dbasis[n * nq + p];
                                
                                    for(unsigned int n = 0; n < nq; ++n)                
                                        tmp0 += s_rqs[b * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
                                
                                    for(unsigned int n = 0; n < nq; ++n)
                                        tmp0 += s_rqt[b * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
                                
                                    d_out[eb * nelmtPerBatch * nq * nq * nq + b  * nq * nq * nq + p * nq * nq + q * nq + r] = tmp0;
                                }
                            }
                            team_member.team_barrier();
                            eb += team_member.league_size();     
                        }
                    }
                );
                Kokkos::fence();
                kokkosTimer.stop();
                const double t_w = kokkosTimer.elapsedSeconds();
                time_kokkos     = std::min(time_kokkos, t_w);
            }

            Kokkos::parallel_reduce(nelmt * nq * nq * nq,
                KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                sum);               

            T gdofPerSeconds = 1.0e-9 * nelmt * nq * nq * nq / time_kokkos;
            results[0] = gdofPerSeconds; 
            results[1] = sum;
            results[2] = time_kokkos;

        }

        return results;
    }


} //namespace Parallel
} //namespace BK5

#endif //BK5_TEMPLATED_KOKKOS_KERNELS_HPP