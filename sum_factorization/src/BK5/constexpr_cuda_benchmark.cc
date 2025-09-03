#include <iostream>
#include <kernels/BK5/cuda_kernels.cuh>
#include <timer.hpp>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

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
    const unsigned int numThreads3D, const unsigned int threadsPerBlockX, 
    const unsigned int threadsPerBlockY, const unsigned int threadsPerBlockZ,
    const unsigned int nelmt, const unsigned int ntests)
{

    //Allocation of arrays
    T* dbasis0 = new T[nq0 * nq0];
    T* dbasis1 = new T[nq1 * nq1];
    T* dbasis2 = new T[nq2 * nq2];
    T* G = new T[nelmt * 6 * nq0 * nq1 * nq2];
    T* in = new T[nelmt * nq0 * nq1 * nq2];
    T* out = new T[nelmt * nq0 * nq1 * nq2];


    //Initialize the input and output arrays
    std::fill(G,   G + nelmt * nq0 * nq1 * nq2 * 6, (T)2.0f);
    std::fill(in,  in + nelmt * nq0 * nq1 * nq2,   (T)3.0f);
    std::fill(out, out + nelmt * nq0 * nq1 * nq2, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int i = 0u; i < nq0; i++)
    {
        for(unsigned int a = 0u; a < nq0; a++)
        {
            dbasis0[i * nq0 + a] = std::cos((T)(i * nq0 + a));
        }
    }
    for(unsigned int j = 0u; j < nq1; j++)
    {
        for(unsigned int b = 0u; b < nq1; b++)
        {
            dbasis1[j * nq1 + b] = std::cos((T)(j * nq1 + b));
        }
    }
    for(unsigned int k = 0u; k < nq2; k++)
    {
        for(unsigned int c = 0u; c < nq2; c++)
        {
            dbasis2[k * nq2 + c] = std::cos((T)(k * nq2 + c));
        }
    }

    T *d_dbasis0, *d_dbasis1, *d_dbasis2, *d_G, *d_in, *d_out;
                              
    CUDA_CHECK(cudaMalloc(&d_dbasis0, nq0 * nq0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis1, nq1 * nq1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis2, nq2 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, nelmt * nq0 * nq1 * nq2 * 6 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nq0 * nq1 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nq0 * nq1 * nq2 * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_dbasis0, dbasis0, nq0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis1, dbasis1, nq1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis2, dbasis2, nq2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, nelmt * nq0 * nq1 * nq2 * 6 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nq0 * nq1 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nq0 * nq1 * nq2 * sizeof(T), cudaMemcpyHostToDevice));

    int device;   cudaGetDevice(&device);   cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // ------------------------- Kernel with 3D block size + Simple Map -------------------------------
    if(nq0 * nq1 * nq2 < prop.maxThreadsPerBlock)
    {   
        const unsigned int numBlocks = numThreads3D / (nq0 * nq1 * nq2);
        unsigned int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_3D_Block_SimpleMap<T><<<numBlocks, nq0 * nq1 * nq2, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D Block Simple Map-> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl;
    }


    // ------------------------- Kernel with 3D block size -------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        const unsigned int numBlocks = numThreads3D / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        unsigned int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_3D_Block<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D Block -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl;
    }


    // ------------------------- Kernel with 2D block size (ij)-------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        const unsigned int numBlocks = (numThreads3D / nq2) / (std::min(nq0 * nq1, threadsPerBlock));
        unsigned int ssize = nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_2D_Block_ij<T><<<numBlocks, std::min(nq0 * nq1, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "2D Block(ij) -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl;
    }


    // ------------------------- Kernel with 2D block size (jk)-------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockY * threadsPerBlockZ;
        const unsigned int numBlocks = (numThreads3D / nq0) / (std::min(nq1 * nq2, threadsPerBlock));
        unsigned int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_2D_Block_jk<T><<<numBlocks, std::min(nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "2D Block(jk) -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl;
    }


    // ------------------------- Kernel with 2D block size (jk) Simple Map-------------------------------
    if(nq1 * nq2 < prop.maxThreadsPerBlock)
    {   
        const unsigned int numBlocks = (numThreads3D / nq0) / (nq1 * nq2);
        unsigned int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_2D_Block_jk_SimpleMap<T><<<numBlocks, nq1 * nq2, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "2D Block(jk) Simple Map -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl;
    }

    cudaFree(d_dbasis0); cudaFree(d_dbasis1); cudaFree(d_dbasis2); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
    delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    constexpr unsigned int nq0                =  4u;
    constexpr unsigned int nq1                =  4u;
    constexpr unsigned int nq2                =  4u;

    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    unsigned int numThreads3D       = (argc > 2) ? atoi(argv[2]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int threadsPerBlockX   = (argc > 3) ? atoi(argv[3]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 4) ? atoi(argv[4]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 5) ? atoi(argv[5]) : nq2;
    unsigned int ntests             = (argc > 6) ? atoi(argv[6]) : 50u;


    std::cout.precision(8);
    run_test<float>(nq0, nq1, nq2, numThreads3D,
                    threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
    
    return 0;
}