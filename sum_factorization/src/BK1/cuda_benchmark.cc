/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <iostream>
#include <kernels/BK1/cuda_kernels.cuh>
#include <timer.hpp>
#include <benchmark_printer.hpp>

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
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int numThreads, const unsigned int threadsPerBlockX, 
    const unsigned int threadsPerBlockY, const unsigned int threadsPerBlockZ,
    const unsigned int nelmt, const unsigned int ntests)
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

    T *d_basis0, *d_basis1, *d_basis2, *d_JxW, *d_in, *d_out;
                              
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


    BenchmarkPrinter printer;
    printer.print_header();


    // ------------------------- Kernel with Warp Centric Computation -------------------------------
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / 2;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        int blockDim(warpSize * nelmtPerBlock);
        int gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_Warp<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }

        printer("WarpCentric", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, gridDim * blockDim, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);
    }

    // ------------------------- Kernel with Warp Centric Computation for Linear Element (Q1) -------------------------------
    if(nm0 == 2 && nm1 == 2 && nm2 == 2)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / 2;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        int blockDim(warpSize * nelmtPerBlock);
        int gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_Warp_Q1<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }

        printer("WarpCentricQ1", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, gridDim * blockDim, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);

    }


    const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

    // ------------------------- Kernel with 1D block size -------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }

        printer("1D", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * (std::min(nq0 * nq1 * nq2, threadsPerBlock)), nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);

    }

    // ------------------------- Kernel with 1D block size + SimpleMap -------------------------------
    {   
        unsigned int threadsPerBlock = nq0 * nq1 * nq2;
        unsigned int numBlocks = numThreads / threadsPerBlock;
        if (numBlocks == 0) numBlocks = 1;

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_SimpleMap<T><<<numBlocks, threadsPerBlock, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        
        printer("1DS", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * threadsPerBlock, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);

    }

    // ------------------------- Kernel with 3D block size -------------------------------
    {
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numBlocks);
        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        printer("3D", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * std::min(nq0 * nq1 * nq2, threadsPerBlock), nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);
    }

    // ------------------------- Kernel with 3D block size + SimpleMap -------------------------------
    {
        const unsigned int threadsPerBlock = nq0 * nq1 * nq2;
        unsigned int numBlocks = numThreads / threadsPerBlock;
        if (numBlocks == 0) numBlocks = 1;

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numBlocks);
        dim3 blockDim(nq0, nq1, nq2);

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        printer("3DS", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * threadsPerBlock, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);
    }

    // ------------------------- Kernel with 2D block size (pq)-------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        unsigned int numBlocks = (numThreads / nq2) / (std::min(nq0 * nq1, threadsPerBlock));
        if (numBlocks == 0) numBlocks = 1;

        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 2 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_2D_BLOCKS_pq<T><<<numBlocks, std::min(nq0 * nq1, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        printer("2D", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * (std::min(nq0 * nq1, threadsPerBlock)), nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);

    }


    // ------------------------- Kernel with 2D block size (pq) + SimpleMap-------------------------------
    {   
        unsigned int numBlocks = (numThreads / nq2) / (nq0 * nq1);
        if (numBlocks == 0) numBlocks = 1;
        
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 2 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK1::Parallel::BwdTransHexKernel_QP_1D_2D_BLOCKS_pq_SimpleMap<T><<<numBlocks, nq0 * nq1, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }

        printer("2DS", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * nq0 * nq1, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);
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
    unsigned int threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    unsigned int ntests             = (argc > 9) ? atoi(argv[9]) : 50u;

    std::cout.precision(8);
    run_test<float>(nq0, nq1, nq2, numThreads,
                    threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);

    return 0;
}