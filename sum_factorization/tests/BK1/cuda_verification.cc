#include <iostream>
#include <kernels/BK1/cuda_kernels.cuh>
#include <kernels/BK1/cuda_mma_kernels.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <cmath>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUDA_LAST_ERROR_CHECK()                                                   \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Last Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

template<typename T>
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int numThreads, const unsigned int threadsPerBlockX,
    const unsigned int threadsPerBlockY, const unsigned int threadsPerBlockZ, const unsigned int nelmt)
{   
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;
    
    //Allocation of arrays
    T* basis0 = new T[nm0 * nq0];
    T* basis1 = new T[nm1 * nq1];
    T* basis2 = new T[nm2 * nq2];
    T* JxW = new T[nelmt * nq0 * nq1 * nq2];
    T* in = new T[nelmt * nm0 * nm1 * nm2];
    T* out = new T[nelmt * nm0 * nm1 * nm2];


    //Initialize the input and output arrays
    std::fill(JxW, JxW + nelmt * nq0 * nq1 * nq2, (T)1.0f);
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (T)3.0f);
    std::fill(out, out + nelmt * nm0 * nm1 * nm2, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq0; p++)
    {
        for(unsigned int i = 0u; i < nm0; i++)
        {
            basis0[p * nm0 + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for(unsigned int q = 0u; q < nq1; q++)
    {
        for(unsigned int j = 0u; j < nm1; j++)
        {
            basis1[q * nm1 + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for(unsigned int r = 0u; r < nq2; r++)
    {
        for(unsigned int k = 0u; k < nm2; k++)
        {
            basis2[r * nm2 + k] = std::cos((T)(r * nm2 + k));
        }
    }




    T  *d_basis0, *d_basis1, *d_basis2, *d_JxW, *d_in, *d_out;

    CUDA_CHECK(cudaMalloc(&d_basis0, nq0 * nm0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis1, nq1 * nm1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis2, nq2 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nm0 * nm1 * nm2 * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_basis0, basis0, nm0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis1, basis1, nm1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis2, basis2, nm2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_JxW, JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));


    // ------------------------- Kernel with Warp Centric Computation -------------------------------
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        dim3 blockDim(warpSize * nelmtPerBlock);
        dim3 gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        BK1::Parallel::BwdTransHexKernel_QP_1D_Warp<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
            
        T resultCuda = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0f,
            thrust::plus<T>());
            
            std::cout << "Cuda Warp-Centric = " << std::sqrt(resultCuda) << std::endl;
    }

    // ------------------------- Kernel with Warp Centric Computation for Linear Element (Q1) Only -------------------------------
    if(nm0 == 2 && nm1 == 2 && nm2 == 2)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        dim3 blockDim(warpSize * nelmtPerBlock);
        dim3 gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        BK1::Parallel::BwdTransHexKernel_QP_1D_Warp_Q1<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
            
        T resultCuda = thrust::transform_reduce(
           thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
           thrust::square<T>(), (T)0.0f,
           thrust::plus<T>());
            
        std::cout << "Cuda Warp-Centric Q1 = " << std::sqrt(resultCuda) << std::endl;
    }

    const unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 2 * nq0 * nq1 * nq2;

    // ------------------------- Cuda kernel with 1D block size --------------------------------------
    {
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        BK1::Parallel::BwdTransHexKernel_QP_1D<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                    d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T resultCuda = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0f,
            thrust::plus<T>());
            
        std::cout << "Cuda 1D blocks norm = " << std::sqrt(resultCuda) << std::endl;
    }

    // ------------------------- Cuda kernel with 1D block size + SimpleMap --------------------------------------
    {
        unsigned int threadsPerBlock = nq0 * nq1 * nq2;
        unsigned int numBlocks = numThreads / threadsPerBlock;
        if (numBlocks == 0) numBlocks = 1;

        BK1::Parallel::BwdTransHexKernel_QP_1D_SimpleMap<T><<<numBlocks, threadsPerBlock, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                    d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T resultCuda = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0f,
            thrust::plus<T>());
            
        std::cout << "Cuda 1D blocks Simple Map norm = " << std::sqrt(resultCuda) << std::endl;
    }

    // ------------------------ Cuda kernel with 3D block size --------------------------------------
    {
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        dim3 gridDim(numBlocks);
        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));
            
        BK1::Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);                       
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
            
        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>());
    
        std::cout << "Cuda 3D blocks norm = " << std::sqrt(result) << std::endl;
    }

    // ------------------------- Cuda kernel with 3D block size + Simple Map --------------------------------------
    {
        const unsigned int threadsPerBlock = nq0 * nq1 * nq2;
        unsigned int numBlocks = numThreads / threadsPerBlock;
        if (numBlocks == 0) numBlocks = 1;

        dim3 blockDim(nq0, nq1, nq2);

        BK1::Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap<T><<<numBlocks, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);                       
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T result = thrust::transform_reduce(
        thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
        thrust::square<T>(), (T)0.0,
        thrust::plus<T>());
            
        std::cout << "Cuda 3D blocks Simple Map norm = " << std::sqrt(result) << std::endl;
    }

    // ------------------------- Kernel with 2D block size (pq)-------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        unsigned int numBlocks = (numThreads / nq2) / (std::min(nq0 * nq1, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 2 * nq0 * nq1 * nq2;          //shared memory dynamic size

        BK1::Parallel::BwdTransHexKernel_QP_1D_2D_BLOCKS_pq<T><<<numBlocks, std::min(nq0 * nq1, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T result = thrust::transform_reduce(
        thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
        thrust::square<T>(), (T)0.0,
        thrust::plus<T>());
            
        std::cout << "Cuda 2D blocks(pq) norm = " << std::sqrt(result) << std::endl;
    }


    // ------------------------- Kernel with 2D block size (pq) + SimpleMap-------------------------------
    {   
        unsigned int numBlocks = (numThreads / nq2) / (nq0 * nq1);
        if (numBlocks == 0) numBlocks = 1;

        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 2 * nq0 * nq1 * nq2;          //shared memory dynamic size

        BK1::Parallel::BwdTransHexKernel_QP_1D_2D_BLOCKS_pq_SimpleMap<T><<<numBlocks, nq0 * nq1, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );
            
        std::cout << "Cuda 2D blocks(pq) SimpleMap norm = " << std::sqrt(result) << std::endl;
    }

    
    // ------------------------- mma Kernel with Single Warp per Element-----------------------------------
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;

        unsigned int numBlocks = numThreads / warpSize;
        if (numBlocks == 0) numBlocks = 1;

        BK1::Parallel::BwdTransHexKernel_mma<double, 8, 8, 4><<<numBlocks, warpSize, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);

        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );
            
        std::cout << "Cuda mma kernel norm = " << std::sqrt(result) << std::endl;
    }

    cudaFree(d_basis0); cudaFree(d_basis1); cudaFree(d_basis2); cudaFree(d_JxW); cudaFree(d_in); cudaFree(d_out);
    delete[] basis0; delete[] basis1; delete[] basis2; delete[] JxW; delete[] in; delete[] out;
}

int main(int argc, char **argv){
    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    unsigned int numThreads         = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;

    unsigned int threadsPerBlockX   = nq0 / 2;
    unsigned int threadsPerBlockY   = nq1 / 2;
    unsigned int threadsPerBlockZ   = nq2 / 2;

    std::cout.precision(8);
    run_test<double>(nq0, nq1, nq2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt);
    
    return 0;
}