/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <iostream>
#include <parallel_kernels.cuh>
#include <serial_kernels.hpp>
#include <timer.hpp>

template<typename T>
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, 
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2, 
    const unsigned int numThreadsX, const unsigned int numThreadsY, const unsigned int numThreadsZ,
    const unsigned int threadsPerBlockX, const unsigned int threadsPerBlockY, const unsigned int threadsPerBlockZ,
    const unsigned int nelmt, const unsigned int ntests)
{
    unsigned int numThreads = numThreadsX * numThreadsY * numThreadsZ;
    unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;

    const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

    //Allocation of arrays
    T* basis0 = new T[nm0 * nq0];
    T* basis1 = new T[nm1 * nq1];
    T* basis2 = new T[nm2 * nq2];
    T* in = new T[nelmt * nm0 * nm1 * nm2];
    T* out = new T[nelmt * nq0 * nq1 * nq2];

    //Initialize the input and output arrays
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (T)3);
    std::fill(out, out + nelmt * nq0 * nq1 * nq2, (T)0);

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

    //Parallel Kernels
    T *d_in, *d_out, *d_basis0, *d_basis1, *d_basis2;

    const unsigned int ssize = nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nm0 * nm1 * nm2          //shared memory dynamic size
                               + nm1 * nm2 * nq0 + nm2 * nq0 * nq1 + nq0 * nq1 * nq2;

    cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T));
    cudaMalloc(&d_out, nelmt * nq0 * nq1 * nq2 * sizeof(T));
    cudaMalloc(&d_basis0, nq0 * nm0 * sizeof(T));
    cudaMalloc(&d_basis1, nq1 * nm1 * sizeof(T));
    cudaMalloc(&d_basis2, nq2 * nm2 * sizeof(T));

    cudaMemcpy(d_in, in, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_basis0, basis0, nm0 * nq0 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_basis1, basis1, nm1 * nq1 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_basis2, basis2, nm2 * nq2 * sizeof(T), cudaMemcpyHostToDevice);

    // ------------------------- Kernel with 1D block size -------------------------------
    {
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nm0, nm1, nm2, nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_in, d_out);
            cudaDeviceSynchronize();
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "1D_Block -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;

        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>());

        std::cout << "1D blocks norm = " << std::sqrt(result) << std::endl;
    }

    // ------------------------- Kernel with 3D block size -------------------------------
    {
        thrust::fill(thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2, static_cast<T>(0));

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numThreadsX / threadsPerBlockX, numThreadsY / threadsPerBlockY, numThreadsZ / threadsPerBlockZ);
        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nm0, nm1, nm2, nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_in, d_out);                       
            cudaDeviceSynchronize();
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D_Block -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;

        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>());

        std::cout << "3D blocks norm = " << std::sqrt(result) << std::endl;
    }

    // ------------------------- Kernel with 3D block size + SimpleMap -------------------------------
    {
        thrust::fill(thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2, static_cast<T>(0));
        
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numThreadsX / nq0, numThreadsY / nq1, numThreadsZ / nq2);
        dim3 blockDim(nq0, nq1, nq2);

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nm0, nm1, nm2, nq0, nq1, nq2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_in, d_out);                       
            cudaDeviceSynchronize();
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << "3D_Block Simple Map -> " << "nelmt = " << nelmt <<" GDoF/s = " << 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time << std::endl;
        
        T result = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq0 * nq1 * nq2,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>());
            
            std::cout << "3D blocks Simple Map norm = " << std::sqrt(result) << std::endl;
    }
            


    cudaFree(d_basis0); cudaFree(d_basis1); cudaFree(d_basis2); cudaFree(d_in); cudaFree(d_out);
    delete[] basis0; delete[] basis1; delete[] basis2; delete[] in, delete[] out;
}


int main(int argc, char **argv){

    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int numThreadsX        = (argc > 4) ? atoi(argv[4]) : 512;
    unsigned int numThreadsY        = (argc > 5) ? atoi(argv[5]) : 512;
    unsigned int numThreadsZ        = (argc > 6) ? atoi(argv[6]) : 512;
    unsigned int threadsPerBlockX   = (argc > 7) ? atoi(argv[7]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 8) ? atoi(argv[8]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 9) ? atoi(argv[9]) : nq2;
    unsigned int nelmt              = (argc > 10) ? atoi(argv[10]) : 2 << 18;
    unsigned int ntests             = (argc > 11) ? atoi(argv[11]) : 100u;


    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    run_test<float>(nq0, nq1, nq2, nm0, nm1, nm2, numThreadsX, numThreadsY, numThreadsZ, 
                    threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);

    return 0;
}