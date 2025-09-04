#include <iostream>
#include <kernels/BK3/cuda_kernels.cuh>
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
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    //Allocation of arrays
    T* basis0 = new T[nm0 * nq0];
    T* basis1 = new T[nm1 * nq1];
    T* basis2 = new T[nm2 * nq2];
    T* dbasis0 = new T[nq0 * nq0];
    T* dbasis1 = new T[nq1 * nq1];
    T* dbasis2 = new T[nq2 * nq2];
    T* G = new T[nelmt * 6 * nq0 * nq1 * nq2];
    T* in = new T[nelmt * nq0 * nq1 * nq2];
    T* out = new T[nelmt * nq0 * nq1 * nq2];


    //Initialize the input and output arrays
    std::fill(G,   G + nelmt * nq0 * nq1 * nq2 * 6, (T)2.0f);
    std::fill(in,  in + nelmt * nm0 * nm1 * nm2,   (T)3.0f);
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

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq0; i++)
    {
        for(unsigned int p = 0u; p < nq0; p++)
        {
            dbasis0[i * nq0 + p] = std::cos((T)(i * nq0 + p));
        }
    }
    for(unsigned int j = 0u; j < nq1; j++)
    {
        for(unsigned int q = 0u; q < nq1; q++)
        {
            dbasis1[j * nq1 + q] = std::cos((T)(j * nq1 + q));
        }
    }
    for(unsigned int k = 0u; k < nq2; k++)
    {
        for(unsigned int r = 0u; r < nq2; r++)
        {
            dbasis2[k * nq2 + r] = std::cos((T)(k * nq2 + r));
        }
    }

    T *d_basis0, *d_basis1, *d_basis2, *d_dbasis0, *d_dbasis1, *d_dbasis2, *d_G, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_basis0, nm0 * nq0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis1, nm1 * nq1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis2, nm2 * nq2 * sizeof(T)));  
    CUDA_CHECK(cudaMalloc(&d_dbasis0, nq0 * nq0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis1, nq1 * nq1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis2, nq2 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, nelmt * nq0 * nq1 * nq2 * 6 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nm0 * nm1 * nm2 * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_basis0, basis0, nm0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis1, basis1, nm1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis2, basis2, nm2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis0, dbasis0, nq0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis1, dbasis1, nq1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis2, dbasis2, nq2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, nelmt * nq0 * nq1 * nq2 * 6 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));

    int device;   cudaGetDevice(&device);   cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    

    // ------------------------- Kernel with 3D block size -------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        const unsigned int numBlocks = numThreads3D / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 5 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK3::Parallel::TransHexKernel_QP_3D_Block<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D Block -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;
    }


    // ------------------------- Kernel with 3D block size + SimpleMap -------------------------------
    {
        const unsigned int numBlocks = numThreads3D / (nq0 * nq1 * nq2);
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 5 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numBlocks);
        dim3 blockDim(nq0, nq1, nq2);

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            BK3::Parallel::TransHexKernel_QP_3D_Block_SimpleMap<T><<<numBlocks, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D_Block Simple Map -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;
    }



    // ------------------------- Kernel with 2D block size (pq)-------------------------------
    {   
        unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        const unsigned int numBlocks = (numThreads3D / nq2) / (std::min(nq0 * nq1, threadsPerBlock));
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 5 * nq0 * nq1 * nq2;          //shared memory dynamic size

    
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK3::Parallel::TransHexKernel_QP_2D_Block_pq<T><<<numBlocks, std::min(nq0 * nq1, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "2D Block(pq) -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;
    }


        // ------------------------- Kernel with 2D block size (pq) + SimpleMap-------------------------------
    {   
        const unsigned int numBlocks = (numThreads3D / nq2) / (nq0 * nq1);
        unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 5 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();

            BK3::Parallel::TransHexKernel_QP_2D_Block_pq_SimpleMap<T><<<numBlocks, nq0 * nq1, ssize * sizeof(T)>>>(nq0, nq1, nq2, nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }

        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );
            
        std::cout << "2D Block(pq) Simple Map -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;
    }
    cudaFree(d_basis0); cudaFree(d_basis1); cudaFree(d_basis2); cudaFree(d_dbasis0); cudaFree(d_dbasis1); cudaFree(d_dbasis2); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
    delete[] basis0; delete[] basis1; delete[] basis2; delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    unsigned int numThreads3D       = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    unsigned int ntests             = (argc > 9) ? atoi(argv[9]) : 50u;


    std::cout.precision(8);
    run_test<float>(nq0, nq1, nq2, numThreads3D,
                    threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
    
    return 0;
}
