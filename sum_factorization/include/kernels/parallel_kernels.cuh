/*
Output-centric approach was used to calculate intermediate memories (group size = output vector size). 
*/

#ifndef PARALLEL_KERNELS_CUH
#define PARALLEL_KERNELS_CUH

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <Kokkos_Core.hpp>
#include <timer.hpp>

namespace Parallel{

template<typename T>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1,
    const unsigned int nq2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nm0 * nm1 * nm2;
    T *s_wsp2 = s_wsp1 + nm1 * nm2 * nq0;
    T *s_wsp3 = s_wsp2 + nm2 * nq0 * nq1;


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

    
    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //Copy inptr to s_wsp0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncthreads();


        // direction 0  -> tid = p * nm1 * nm2 + j * nm2 + k (wsp1)
        for(unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2; tid += blockDim.x)
        {
            unsigned int p = tid / (nm1 * nm2);
            unsigned int j = (tid % (nm1 * nm2)) / nm2;
            unsigned int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();


        //direction 1 -> tid = q * nq0 * nm2 + p * nm2 + k  (wsp2)
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
        {
            unsigned int q = tid / (nq0 * nm2);
            unsigned int p = (tid % (nq0 * nm2)) / nm2;
            unsigned int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp2[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncthreads();


        //direction 2 -> tid = p * nq1 * nq2 + q * nq2 + r   (wsp3)
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x)
        {
            unsigned int p = tid / (nq1 * nq2);
            unsigned int q = (tid % (nq1 * nq2)) / nq2;
            unsigned int r = tid % nq2;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp2[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp3[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        __syncthreads();


        //Copy s_wsp3 to outptr
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x)
        {
            d_out[e * nq0 * nq1 * nq2 + tid] = s_wsp3[tid];
        } 
        __syncthreads();

        e += gridDim.x;
    }
}

template <typename T>
T KokkosKernel(const unsigned int nelmt, const unsigned int ntests, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int numThreads, const unsigned int threadsPerBlock)
    {   
        const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

        T result_kokkos = 0.0;
        Kokkos::initialize();
        {
            Kokkos::View<T*> d_in("d_in", nelmt * nm0 * nm1 * nm2);
            Kokkos::View<T*> d_out("d_out", nelmt * nq0 * nq1 * nq2);
            Kokkos::View<T*> d_basis0("d_basis0", nm0 * nq0);
            Kokkos::View<T*> d_basis1("d_basis1", nm1 * nq1);
            Kokkos::View<T*> d_basis2("d_basis2", nm2 * nq2);
            
            Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, nelmt),
            KOKKOS_LAMBDA (int elmtIdx){
                for(int i=0; i < nm0 * nm1 * nm2; ++i)
                    d_in(elmtIdx * nm0 * nm1 * nm2 + i) = (T)3;
                
                for(int i=0; i < nq0 * nq1 * nq2; ++i)
                    d_out(elmtIdx * nq0 * nq1 * nq2 + i) = (T)0;
                });
                
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0u, 0u}, {nq0, nm0}),
                KOKKOS_LAMBDA(const unsigned p, const unsigned i) {
                    d_basis0(p * nm0 + i) = std::cos((T)(p * nm0 + i));
                });
                    
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0u, 0u}, {nq1, nm1}),
                KOKKOS_LAMBDA(const unsigned q, const unsigned j) {
                    d_basis1(q * nm1 + j) = std::cos((T)(q * nm1 + j));
                });
                    
            Kokkos::parallel_for(
                Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<2>>({0u, 0u}, {nq2, nm2}),
                KOKKOS_LAMBDA(const unsigned r, const unsigned k) {
                    d_basis2(r * nm2 + k) = std::cos((T)(r * nm2 + k));
                });

            Kokkos::fence();

            Timer kokkosTimer;
            double time_kokkos = std::numeric_limits<T>::max();

            for (unsigned int t = 0u; t < ntests; ++t)
            {
                kokkosTimer.start();
                //Kokkos with shared memory
                const unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nm0 * nm1 * nm2          //shared memory dynamic size
                                         + nm1 * nm2 * nq0 + nm2 * nq0 * nq1 + nq0 * nq1 * nq2;

                const unsigned int shmem_size = Kokkos::View<T *, Kokkos::DefaultExecutionSpace::scratch_memory_space,
                                                Kokkos::MemoryTraits<Kokkos::Unmanaged>>::shmem_size(ssize);
                
                typedef Kokkos::TeamPolicy<Kokkos::Cuda>::member_type member_type;
                Kokkos::TeamPolicy<Kokkos::Cuda> policy(numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock));
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
                        T *s_wsp1 = s_wsp0 + nm0 * nm1 * nm2;
                        T *s_wsp2 = s_wsp1 + nm1 * nm2 * nq0;
                        T *s_wsp3 = s_wsp2 + nm2 * nq0 * nq1;
                    

                        unsigned int threadIdx = team_member.team_rank();
                        unsigned int blockSize = team_member.team_size();
                        for(unsigned int tid = threadIdx; tid < nm0 * nq0; tid += blockSize)
                            s_basis0[tid] = d_basis0(tid);
                        for(unsigned int tid = threadIdx; tid < nm1 * nq1; tid += blockSize)
                            s_basis1[tid] = d_basis1(tid);
                        for(unsigned int tid = threadIdx; tid < nm2 * nq2; tid += blockSize)
                            s_basis2[tid] = d_basis2(tid);
                    
                        while(e < nelmt)
                        {   
                            //Copy d_in to s_wsp0
                            for(unsigned int tid = threadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
                            {
                                s_wsp0[tid] = d_in(e * nm0 * nm1 * nm2 + tid);
                            }
                            team_member.team_barrier();

                            // direction 0 -> i = p * nm1 * nm2 + j * nm2 + k (wsp1)
                            for(unsigned int tid = threadIdx; tid < nq0 * nm1 * nm2; tid += blockSize)
                            {
                                unsigned int p = tid / (nm1 * nm2);
                                unsigned int j = (tid % (nm1 * nm2)) / nm2;
                                unsigned int k =  tid % nm2;

                                T tmp = 0.0;
                                for(unsigned int i = 0; i < nm0; ++i)
                                {
                                    tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                                }

                                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();

                            //direction 1 -> tid = q * nq0 * nm2 + p * nm2 + k  (wsp2)
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nm2; tid += blockSize)
                            {
                                unsigned int q = tid / (nq0 * nm2);
                                unsigned int p = (tid % (nq0 * nm2)) / nm2;
                                unsigned int k = tid % nm2;

                                T tmp = 0.0;
                                for(unsigned int j = 0; j < nm1; j++)
                                {
                                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                                }
                                s_wsp2[q * nq0 * nm2 + p * nm2 + k] = tmp;
                            }
                            team_member.team_barrier();

                            //direction 2 -> tid = p * nq1 * nq2 + q * nq2 + r   (wsp3)
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2; tid += blockSize)
                            {
                                unsigned int p = tid / (nq1 * nq2);
                                unsigned int q = (tid % (nq1 * nq2)) / nq2;
                                unsigned int r = tid % nq2;

                                T tmp = 0.0;
                                for(unsigned int k = 0; k < nm2; ++k)
                                {
                                    tmp += s_wsp2[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                                }
                                s_wsp3[p * nq1 * nq2 + q * nq2 + r] = tmp;
                            }
                            team_member.team_barrier();

                            //Copy s_wsp3 to outptr
                            for(unsigned int tid = threadIdx; tid < nq0 * nq1 * nq2; tid += blockSize)
                            {
                                d_out(e * nq0 * nq1 * nq2 + tid) = s_wsp3[tid];
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
            std::cout << "Kokkos -> " << "nelmt = " << nelmt << " GDoF/s = " 
                      << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time_kokkos << std::endl;

            

            Kokkos::parallel_reduce(nelmt * nq0 * nq1 * nq2,
                    KOKKOS_LAMBDA(unsigned int i, T &val) {
                    val += d_out(i) * d_out(i);
                },
                result_kokkos);
            
            }
        Kokkos::finalize();
        return result_kokkos;
    }


    // In 3D thread-blocks in CUDA, X dimension is fastest, Z is slowest
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy

    template <typename T>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS(
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nq0, const unsigned int nq1,
        const unsigned int nq2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nm0 * nm1 * nm2;
        T *s_wsp2 = s_wsp1 + nm1 * nm2 * nq0;
        T *s_wsp3 = s_wsp2 + nm2 * nq0 * nq1;
    
        //Finding global indices
        unsigned int totalThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        unsigned int blockThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

        //copy to shared memory
        for(unsigned int tid = blockThreadIdx; tid < nq0 * nm0; tid += totalThreadsPerBlock)
        {
            s_basis0[tid] = d_basis0[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq1 * nm1; tid += totalThreadsPerBlock)
        {
            s_basis1[tid] = d_basis1[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq2 * nm2; tid += totalThreadsPerBlock)
        {
            s_basis2[tid] = d_basis2[tid];
        }
    
        
        unsigned int e = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    
        while(e < nelmt)
        {
            //Copy inptr to s_wsp0
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += totalThreadsPerBlock)
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            __syncthreads();
    
    
            // direction 0  -> tid = p * nm1 * nm2 + j * nm2 + k (wsp1)
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){
                        if(threadIdx.y < nm1 && threadIdx.z < nm2){
                            T tmp = 0.0;
                            for(unsigned int i = 0; i < nm0; ++i)
                            {
                                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                            }
                            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
                        }
                    }
                }
            }
            __syncthreads();
    
    
            //direction 1 -> tid = q * nq0 * nm2 + p * nm2 + k  (wsp2)
            for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){
                        if(threadIdx.z < nm2){                  
                            T tmp = 0.0;
                            for(unsigned int j = 0; j < nm1; j++)
                            {
                                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                            }
                            s_wsp2[q * nq0 * nm2 + p * nm2 + k] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
    

            //direction 2 -> tid = p * nq1 * nq2 + q * nq2 + r   (wsp3)
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                    for(unsigned int r = threadIdx.z; r < nq2; r += blockDim.z){ 
                        T tmp = 0.0;
                        for(unsigned int k = 0; k < nm2; ++k)
                        {
                            tmp += s_wsp2[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                        }
                        s_wsp3[p * nq1 * nq2 + q * nq2 + r] = tmp;
                    }
                }
            }
            __syncthreads();
    
    
            //Copy s_wsp3 to outptr
            for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += totalThreadsPerBlock)
            {
                d_out[e * nq0 * nq1 * nq2 + tid] = s_wsp3[tid];
            } 
            __syncthreads();
    
            e += gridDim.x * gridDim.y * gridDim.z;
        }
    }


    template <typename T>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap(
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nq0, const unsigned int nq1,
        const unsigned int nq2, const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nm0 * nm1 * nm2;
        T *s_wsp2 = s_wsp1 + nm1 * nm2 * nq0;
        T *s_wsp3 = s_wsp2 + nm2 * nq0 * nq1;
    
        //Finding global indices
        unsigned int totalThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        unsigned int blockThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

        //copy to shared memory
        for(unsigned int tid = blockThreadIdx; tid < nq0 * nm0; tid += totalThreadsPerBlock)
        {
            s_basis0[tid] = d_basis0[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq1 * nm1; tid += totalThreadsPerBlock)
        {
            s_basis1[tid] = d_basis1[tid];
        }
    
        for(unsigned int tid = blockThreadIdx; tid < nq2 * nm2; tid += totalThreadsPerBlock)
        {
            s_basis2[tid] = d_basis2[tid];
        }
    
        unsigned int e = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
    
        while(e < nelmt)
        {   
            unsigned int p, q, r, j, k;
            //Copy inptr to s_wsp0
            for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += totalThreadsPerBlock)
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            __syncthreads();
    
    
            // direction 0  -> tid = p * nm1 * nm2 + j * nm2 + k (wsp1)

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
    
    
            //direction 1 -> tid = q * nq0 * nm2 + p * nm2 + k  (wsp2)

            if(threadIdx.y < nq1 && threadIdx.x < nq0 && threadIdx.z < nm2){ 
                q = threadIdx.y;
                p = threadIdx.x;
                k = threadIdx.z;
                
                T tmp = 0.0;
                for(unsigned int j = 0; j < nm1; j++)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                }
                s_wsp2[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            __syncthreads();
    

            //direction 2 -> tid = p * nq1 * nq2 + q * nq2 + r   (wsp3)
            if(threadIdx.x < nq0 && threadIdx.y < nq1  && threadIdx.z < nq2){ 
                p = threadIdx.x;
                q = threadIdx.y;
                r = threadIdx.z;
                T tmp = 0.0;
                for(unsigned int k = 0; k < nm2; ++k)
                {
                    tmp += s_wsp2[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                }
                s_wsp3[p * nq1 * nq2 + q * nq2 + r] = tmp;
            }
            __syncthreads();
    

            //Copy s_wsp3 to outptr
            for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += totalThreadsPerBlock)
            {
                d_out[e * nq0 * nq1 * nq2 + tid] = s_wsp3[tid];
            } 
            __syncthreads();
    
            e += gridDim.x * gridDim.y * gridDim.z;
        }
    }
} //namespace Parallel


#endif //PARALLEL_KERNELS_CUH