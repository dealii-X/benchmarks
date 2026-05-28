#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "mma_f64_m8n8k4_MassOperator.cuh"
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


template<typename T, const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nelmtPerBatch>
void run_test(const unsigned int nelmt, const unsigned int ntests)
{   
    constexpr unsigned int ndof_1D = nm_n * nm_t * nm_t;
    constexpr unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;
    
    const unsigned int padded_nelmt = ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch) * nelmtPerBatch;
    constexpr unsigned int threadsPerBlock = 32; 
    const unsigned int numBlocks = std::max(1U, (padded_nelmt / nelmtPerBatch));

    size_t shmem_size = (nm_n * nq + nm_t * nq + 5 * nelmtPerBatch * nq * nq * nq) * sizeof(T);



    // --- Host Allocations ---
    T* basis_n = new T[nm_n * nq];
    T* basis_t = new T[nm_t * nq];

    T* G   = new T[(size_t)padded_nelmt * 6 * nq * nq * nq];
    T* in  = new T[(size_t)padded_nelmt * ndof_total];
    T* out = new T[(size_t)padded_nelmt * ndof_total];

    std::fill(in,  in + (size_t)padded_nelmt * ndof_total, (T)3.0f);
    std::fill(out, out + (size_t)padded_nelmt * ndof_total, (T)0.0f);

    for(size_t i = 0u; i < (size_t)padded_nelmt * 6 * nq * nq * nq; i++) {
        G[i] = std::cos((T)i);
    }

    for(unsigned int i = 0u; i < nm_n; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis_n[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    for(unsigned int i = 0u; i < nm_t; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis_t[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    // --- Device Allocations ---
    T *d_basis_n, *d_basis_t, *d_G, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_basis_n, nm_n * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis_t, nm_t * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G,  (size_t)padded_nelmt * 6 * nq * nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)padded_nelmt * ndof_total * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out,(size_t)padded_nelmt * ndof_total * sizeof(T)));

    // --- Device Transfers ---
    CUDA_CHECK(cudaMemcpy(d_basis_n, basis_n, nm_n * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis_t, basis_t, nm_t * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G,  G,  (size_t)padded_nelmt * 6 * nq * nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, (size_t)padded_nelmt * ndof_total * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out,out,(size_t)padded_nelmt * ndof_total * sizeof(T), cudaMemcpyHostToDevice));


    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- Mass Operator Kernel Launch --------------------------------------
    double time = std::numeric_limits<double>::max();
    Timer timer;

    for (unsigned int t = 0u; t < ntests; ++t)
    {   
        timer.start();
        
        Parallel::f64_m8n8k4_Mass_mma<nq, nm_t, nm_n, nelmtPerBatch><<<numBlocks, threadsPerBlock, shmem_size>>>(
                padded_nelmt, numBlocks, threadsPerBlock, d_basis_n, d_basis_t, d_G, d_in, d_out, ntests);

        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
        
        timer.stop();
        time = std::min(time, timer.elapsedSeconds());
    }

    CUDA_CHECK(cudaMemcpy(out, d_out, (size_t)padded_nelmt * ndof_total * sizeof(T), cudaMemcpyDeviceToHost));

    T sum = 0;
    for(size_t i = 0; i < (size_t)nelmt * ndof_total; i++) {
        sum += out[i] * out[i];
    }

    uint64_t nDOF  = (uint64_t)ndof_total * nelmt; 
    uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
    T bw = 1.0e-9 * (2 * nDOF + 6 * nQuad) * sizeof(T) / time;

    T DOFs = 1.0e-9 * nDOF / time;

    printer("MMA_fp64", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, (double)DOFs, bw, std::sqrt(sum));

    delete[] basis_n; delete[] basis_t; delete[] G; delete[] in; delete[] out;
    cudaFree(d_basis_n); cudaFree(d_basis_t); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
}



int main(int argc, char **argv){

    using T = double;
    constexpr size_t shmemPerBlock = 48000;

    unsigned int p         = (argc > 1) ? atoi(argv[1]) : 2u;    unsigned int nq     = p + 2;
    unsigned int nelmt     = (argc > 2) ? atoi(argv[2]) : 2 << 15;
    unsigned int ntests    = (argc > 3) ? atoi(argv[3]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 3:  run_test<T, 3,  2, 3,  std::max(1UL, shmemPerBlock / (5 * 3*3*3) / sizeof(T))>(nelmt, ntests); break;
        case 4:  run_test<T, 4,  3, 4,  std::max(1UL, shmemPerBlock / (5 * 4*4*4) / sizeof(T))>(nelmt, ntests); break;
        case 5:  run_test<T, 5,  4, 5,  std::max(1UL, shmemPerBlock / (5 * 5*5*5) / sizeof(T))>(nelmt, ntests); break;
        case 6:  run_test<T, 6,  5, 6,  std::max(1UL, shmemPerBlock / (5 * 6*6*6) / sizeof(T))>(nelmt, ntests); break;
        case 7:  run_test<T, 7,  6, 7,  std::max(1UL, shmemPerBlock / (5 * 7*7*7) / sizeof(T))>(nelmt, ntests); break;
        case 8:  run_test<T, 8,  7, 8,  std::max(1UL, shmemPerBlock / (5 * 8*8*8) / sizeof(T))>(nelmt, ntests); break;
        case 9:  run_test<T, 9,  8, 9,  std::max(1UL, shmemPerBlock / (5 * 9*9*9) / sizeof(T))>(nelmt, ntests); break;
        case 10: run_test<T, 10, 9, 10, std::max(1UL, shmemPerBlock / (5 * 10*10*10) / sizeof(T))>(nelmt, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    return 0;
}