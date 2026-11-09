#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <kernels/BK4/templated_cuda_kernels.cuh>
#include <timer.hpp>
#include <benchmark_printer.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "\
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_LAST_ERROR_CHECK()                                               \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Last Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;
    const unsigned int ncomp = 3; // 3D Vector field has 3 components

    // Allocation of arrays (cast to size_t to prevent overflow for large nelmt)
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];

    T* G = new T[(size_t)nelmt * 6 * nquad];
    T* in = new T[(size_t)nelmt * ncomp * nm * nm * nm];
    T* out = new T[(size_t)nelmt * ncomp * nm * nm * nm];

    // Initialize the input and output arrays (3 components for vector field)
    for(size_t i = 0; i < (size_t)nelmt * ncomp * nm * nm * nm; ++i)
        in[i] = std::sin((T)i); 

    std::fill(out, out + (size_t)nelmt * ncomp * nm * nm * nm, (T)0.0);

    // Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[i * nq + p] = std::sin((T)(i * nq + p));
        }
    }

    // Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    //initialize G
    for(size_t i = 0u; i < (size_t)nelmt * 6 * nquad; i++)
    {
        G[i] = std::cos((T)(i));
    }

    // Allocate and transfer device memory
    T *d_basis, *d_dbasis, *d_G, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_basis, nm * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis, nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, (size_t)nelmt * 6 * nquad * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)nelmt * ncomp * nm * nm * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, (size_t)nelmt * ncomp * nm * nm * nm * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_basis, basis, nm * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis, dbasis, nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, (size_t)nelmt * 6 * nquad * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, (size_t)nelmt * ncomp * nm * nm * nm * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, (size_t)nelmt * ncomp * nm * nm * nm * sizeof(T), cudaMemcpyHostToDevice));

    BenchmarkPrinter<T> printer;

    // ------------------------- BK4 Vector Laplace Operator Kernel ------------------------------------
    {
        //set max shmem per block
        size_t shmem_bytes = 225 * 1000; // 227 KB for Hopper

        cudaFuncSetAttribute(
            BK4::Parallel::LaplaceOperator<T, nq>, 
            cudaFuncAttributeMaxDynamicSharedMemorySize, 
            shmem_bytes
        );
        unsigned int ssize = nm * nq + nq * nq + 4 * nelmtPerBatch * nq * nq * nq; 

        double time = std::numeric_limits<double>::max();
        Timer timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            timer.start();
            BK4::Parallel::LaplaceOperator<T, nq><<<numBlocks, threadsPerBlock, ssize * sizeof(T)>>>(
                nelmt, nelmtPerBatch, d_basis, d_dbasis, d_G, d_in, d_out
            );
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            time = std::min(time, timer.elapsedSeconds());
        }
        
        T sum = thrust::transform_reduce(
            thrust::device, d_out, d_out + (size_t)nelmt * ncomp * nm * nm * nm,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );
        
        uint64_t nDOF = (uint64_t)ncomp * nm * nm * nm * nelmt; 
        uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        
        T DOFs = 1.0e-9 * nDOF / time;
        T bw = 1.0e-9 * (2 * nDOF + 6 * nQuad) * sizeof(T) / time; // Corrected to 2 * nDOF
        printer("BK4", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    CUDA_CHECK(cudaFree(d_basis)); 
    CUDA_CHECK(cudaFree(d_dbasis)); 
    CUDA_CHECK(cudaFree(d_G)); 
    CUDA_CHECK(cudaFree(d_in)); 
    CUDA_CHECK(cudaFree(d_out));

    delete[] basis; delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    BenchmarkPrinter<T> printer;
    printer.print_header();

    constexpr int NumSample = 20;
    constexpr double DOFmin = 1e4;
    constexpr double DOFmax = 1e8;

    const double log_step = std::pow(DOFmax / DOFmin, 1.0 / (NumSample - 1));
    const unsigned int ntests = 10;

    for (int istep = 0; istep < NumSample; ++istep)
    {
        size_t dof = static_cast<size_t>(std::llround(DOFmin * std::pow(log_step, istep)));

        for (int nq = 3; nq <= 18; ++nq) {
            int nm = nq - 1;

            // 3D vector field has 3 components, dividing accordingly
            size_t nelmt = dof / (3 * nm * nm * nm);
            if (nelmt == 0) continue;

            unsigned int nelmtPerBatch = std::max(1UL, shmemPerBlock / (4 * nq * nq * nq) / sizeof(T));
            unsigned int numBlocks = std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);
            unsigned int threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);

            switch (nq) {
                case 3:  run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 4:  run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 5:  run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 6:  run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 7:  run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 8:  run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 9:  run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 10: run_test<T, 10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 11: run_test<T, 11>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 12: run_test<T, 12>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 13: run_test<T, 13>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 14: run_test<T, 14>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 15: run_test<T, 15>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 16: run_test<T, 16>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 17: run_test<T, 17>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
                case 18: run_test<T, 18>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;

                default: break;
            }
        }
    }

    return 0;
}