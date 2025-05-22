/*
Output-centric approach was used to calculate intermediate memories (group size = output vector size). 
*/

#ifndef PARALLEL_KERNELS_CUH
#define PARALLEL_KERNELS_CUH

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <vector>

namespace Parallel{

template<typename T>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;



    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    int i, j, k, p, q, r;
    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncthreads();


        //step-2 : direction 0
        for(unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2; tid += blockDim.x)
        {
            p = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();


        //step-3 : direction 1
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
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
        __syncthreads();


        //step-4 : direction 2
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x)
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
        __syncthreads();

        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x){
            s_wsp0[tid] = (T)0;
        }
        
        
        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x){
            s_wsp1[tid] *= JxW[e * nq0 * nq1 * nq2 + tid];
        }
        __syncthreads();

        //step-6 : direction 2
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
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
        __syncthreads();

        //step-7 : direction 1
        for(unsigned int tid = threadIdx.x; tid < nm1 * nm2 * nq0; tid += blockDim.x)
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
        __syncthreads();

        //step-8 : direction 0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
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
        __syncthreads();

        //step-9 : Copy wsp0 to out
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        __syncthreads();

        e += gridDim.x;
    }
}


template <typename T>
std::vector<T> KokkosKernel(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2, const T *__restrict__ basis0, 
    const T *__restrict__ basis1, const T *__restrict__ basis2, const T* __restrict__ JxW, const T* __restrict__ in, const T* __restrict__ out,
    const unsigned int numThreads, const unsigned int threadsPerBlock, const unsigned int nelmt, const unsigned int ntests)
    {   
        const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

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
            Kokkos::deep_copy(d_out, out_view);

            Kokkos::fence();

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            for (unsigned int t = 0u; t < ntests; ++t)
            {
                kokkosTimer.start();
                //Kokkos with shared memory
                const unsigned int ssize = 2 * nq0 + nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;         

                const unsigned int shmem_size = Kokkos::View<T *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
                
                typedef Kokkos::TeamPolicy<>::member_type member_type;
                Kokkos::TeamPolicy<> policy(numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock));
                policy.set_scratch_size(0, Kokkos::PerTeam(ssize * sizeof(T)));

                Kokkos::parallel_for(policy,
                    KOKKOS_LAMBDA (member_type team_member){
                        //element index
                        unsigned int e = team_member.league_rank();

                        //shared memory access
                        T* scratch = (T*)team_member.team_shmem().get_shmem(shmem_size * sizeof(T));
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
                            
                            for(unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2; tid += blockSize){
                                s_wsp0[tid] = (T)0;
                            }
                            
                            //Reverse Operations
                            
                            //step-5 : Multiply with weights and determinant of Jacobi
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
                                s_wsp1[tid] *= d_JxW[tid];
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


    // In 3D thread-blocks in CUDA, X dimension is fastest, Z is slowest
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
    // threadIdx.x for i and p
    // threadIdx.x for j and q
    // threadIdx.x for k and r

    template <typename T>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS(
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
        const unsigned int nm0, const unsigned int nm1,
        const unsigned int nm2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    
        //Finding global indices
        unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
        unsigned int blockThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

        //copy to shared memory
        for(unsigned int tid = blockThreadIdx; tid < nq0 * nm0; tid += blockSize)
        {
            s_basis0[tid] = d_basis0[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq1 * nm1; tid += blockSize)
        {
            s_basis1[tid] = d_basis1[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq2 * nm2; tid += blockSize)
        {
            s_basis2[tid] = d_basis2[tid];
        }
    
    
    
        
        unsigned int e = blockIdx.x;
    
        while(e < nelmt)
        {
            //step-1 : Copy from in to the wsp0
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            __syncthreads();
    
    
            //step-2 : direction 0
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int i = 0; i < nm0; ++i)
                        {
                            tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                        }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
                        
                    }
                }
            }
            __syncthreads();
    
    
            //step-3 : direction 1
            for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int j = 0; j < nm1; j++)
                        {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                        }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    

            //step-4 : direction 2
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                    for(unsigned int r = threadIdx.z; r < nq2; r += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int k = 0; k < nm2; ++k)
                        {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                        }
                        s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
                    }
                }
            }
            __syncthreads();
    
            
            for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x){
                s_wsp0[tid] = (T)0;
            }
            __syncthreads();
            
            //Reverse Operations
    
            //step-5 : Multiply with weights and determinant of Jacobi
            for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
                s_wsp1[tid] *= d_JxW[e * nq0 * nq1 * nq2 + tid];
            }
            __syncthreads();
    
            //step-6 : direction 2
            for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){
                    
                        T tmp = 0.0;
                        for(unsigned int r = 0; r < nq2; ++r)
                        {
                            tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                        }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    
            //step-7 : direction 1
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){    

                        T tmp = 0.0;
                        for(unsigned int q = 0; q < nq1; q++)
                        {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                        }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    
            //step-8 : direction 0
            for(unsigned int i = threadIdx.x; i < nm0; i += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int p = 0; p < nq0; ++p)
                        {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                        }
                        s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;       
                    }
                }
            }
            __syncthreads();


            //step-9 : Copy wsp0 to out
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
            } 
            __syncthreads();
    
            e += gridDim.x;
        }
    }


    template <typename T>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap(
        const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
        const unsigned int nm0, const unsigned int nm1,
        const unsigned int nm2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    
        //Finding global indices
        unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
        unsigned int blockThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        
        //copy to shared memory
        for(unsigned int tid = blockThreadIdx; tid < nq0 * nm0; tid += blockSize)
        {
            s_basis0[tid] = d_basis0[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq1 * nm1; tid += blockSize)
        {
            s_basis1[tid] = d_basis1[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq2 * nm2; tid += blockSize)
        {
            s_basis2[tid] = d_basis2[tid];
        }
        
        
        unsigned int p, q, r, i, j, k;
        unsigned int e = blockIdx.x;
        
        while(e < nelmt)
        {   

            //step-1 : Copy from in to the wsp0
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            __syncthreads();
    
            //step-2 : direction 0
            if(threadIdx.x < nq0 && threadIdx.y < nm1 && threadIdx.z < nm2){
                p = threadIdx.x;
                j = threadIdx.y;
                k = threadIdx.z;

                T tmp = 0.0;
                for(unsigned int i = 0; i < nm0; ++i)
                {
                    tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                }
                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
            }
            __syncthreads();
    
            //step-3 : direction 1
            if(threadIdx.y < nq1 && threadIdx.x < nq0 && threadIdx.z < nm2){ 

                q = threadIdx.y;
                p = threadIdx.x;
                k = threadIdx.z;
                
                T tmp = 0.0;
                for(unsigned int j = 0; j < nm1; j++)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                }
                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            __syncthreads();
    

            //step-4 : direction 2
            if(threadIdx.x < nq0 && threadIdx.y < nq1  && threadIdx.z < nq2){ 
                p = threadIdx.x;
                q = threadIdx.y;
                r = threadIdx.z;

                T tmp = 0.0;
                for(unsigned int k = 0; k < nm2; ++k)
                {
                    tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                }
                s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
            }
            __syncthreads();
            
            //Reverse Operations
            for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
                s_wsp0[tid] = (T)0.0f;
            }
         
            //step-5 : Multiply with weights and determinant of Jacobi
            for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
                s_wsp1[tid] *= d_JxW[e * nq0 * nq1 * nq2 + tid];
            }
            __syncthreads();
            

            //step-6 : direction 2
            if(threadIdx.x < nq0 && threadIdx.y < nq1  && threadIdx.z < nm2)
            {
                p = threadIdx.x;
                q = threadIdx.y;
                k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int r = 0; r < nq2; ++r)
                {
                    tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                }
                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-7 : direction 1
            if(threadIdx.x < nq0 && threadIdx.y < nm1  && threadIdx.z < nm2)
            {
                p = threadIdx.x;
                j = threadIdx.y;
                k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int q = 0; q < nq1; q++)
                {
                    tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                }
                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-8 : direction 0
            if(threadIdx.x < nm0 && threadIdx.y < nm1 && threadIdx.z < nm2)
            {
                i = threadIdx.x;
                j = threadIdx.y;
                k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int p = 0; p < nq0; ++p)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                }
                s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-9 : Copy wsp0 to out
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
            } 
            __syncthreads();
    
            e += gridDim.x;
        }
    }
} //namespace Parallel


#endif //PARALLEL_KERNELS_CUH