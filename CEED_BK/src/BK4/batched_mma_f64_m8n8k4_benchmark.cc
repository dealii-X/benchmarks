#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

#include <kernels/BK4/mma_f64_m8n8k4.cuh>
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


template<typename T, const unsigned int nq, const unsigned int nm, const unsigned int nelmtPerBatch>
void run_test(const unsigned int nelmt, const unsigned int ntests)
{   
    constexpr unsigned int ndof_1D = nm * nm * nm;
    constexpr unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;
    
    const unsigned int padded_nelmt = ((nelmt + nelmtPerBatch - 1) / nelmtPerBatch) * nelmtPerBatch;
    const unsigned int numBlocks = std::max(1U, (padded_nelmt / nelmtPerBatch));

    const unsigned int total_m_tiles = (nelmtPerBatch * nm * nm + 7) / 8;
    
    const unsigned int num_warps = std::min(32U, std::max(1U, total_m_tiles));
    const unsigned int threadsPerBlock = num_warps * 32;

    size_t shmem_size = (nm * nq + nq * nq + 4 * nelmtPerBatch * nq * nq * nq) * sizeof(T);



    // --- Host Allocations ---
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];

    T* G   = new T[(size_t)padded_nelmt * 6 * nq * nq * nq];
    T* in  = new T[(size_t)padded_nelmt * ndof_total];
    T* out = new T[(size_t)padded_nelmt * ndof_total];

    //Initialize the input and output arrays
    for(unsigned int i = 0; i < nelmt * ndof_total; ++i)
        in[i] = std::sin(i);
    
    std::fill(out, out + (size_t)padded_nelmt * ndof_total, (T)0.0f);

    for(size_t i = 0u; i < (size_t)padded_nelmt * 6 * nq * nq * nq; i++) {
        G[i] = std::cos(i);
    }

    for(unsigned int i = 0u; i < nm; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis[i * nq + p] = std::sin((i * nq + p));
        }
    }

    for(unsigned int i = 0u; i < nq; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            dbasis[i * nq + p] = std::cos((i * nq + p));
        }
    }

    // --- Device Allocations ---
    T *d_basis, *d_dbasis, *d_G, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_basis, nm * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis, nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G,  (size_t)padded_nelmt * 6 * nq * nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, (size_t)padded_nelmt * ndof_total * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out,(size_t)padded_nelmt * ndof_total * sizeof(T)));

    // --- Device Transfers ---
    CUDA_CHECK(cudaMemcpy(d_basis, basis, nm * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis, dbasis, nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G,  G,  (size_t)padded_nelmt * 6 * nq * nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, (size_t)padded_nelmt * ndof_total * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out,out,(size_t)padded_nelmt * ndof_total * sizeof(T), cudaMemcpyHostToDevice));

    // ------------------------- Mass Operator Kernel Launch --------------------------------------

    //set max shmem per block
    size_t shmem_bytes = 99 * 1000; // 227 KB for Hopper

    cudaFuncSetAttribute(
        Parallel::f64_m8n8k4_mma<nq, nm, nelmtPerBatch>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shmem_bytes
    );

    double time = std::numeric_limits<double>::max();
    Timer timer;

    for (unsigned int t = 0u; t < ntests; ++t)
    {   
        timer.start();
        
        Parallel::f64_m8n8k4_mma<nq, nm, nelmtPerBatch><<<numBlocks, threadsPerBlock, shmem_size>>>(
                padded_nelmt, d_basis, d_dbasis, d_G, d_in, d_out, ntests);

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
    T bw = 1.0e-9 * (6 * nDOF + 6 * nQuad) * sizeof(T) / time;

    T DOFs = 1.0e-9 * nDOF / time;

    BenchmarkPrinter<T> printer;
    printer("MMA_fp64", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, (double)DOFs, bw, std::sqrt(sum));

    delete[] basis; delete[] dbasis; delete[] G; delete[] in; delete[] out;
    cudaFree(d_basis); cudaFree(d_dbasis); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
}


int main(int argc, char **argv){

    using T = double;
    constexpr size_t shmemPerBlock = 10000;

    // Optional argument for number of tests, defaulting to 10 to match Kokkos code
    unsigned int ntests = (argc > 1) ? atoi(argv[1]) : 10u;

    std::cout.precision(8);
    
    BenchmarkPrinter<T> printer;
    printer.print_header();

    constexpr int NumSample = 20;
    constexpr double DOFmin = 1e4;
    constexpr double DOFmax = 1e8;

    const double log_step = std::pow(DOFmax / DOFmin, 1.0 / (NumSample - 1));

    for (int istep = 0; istep < NumSample; ++istep)
    {
        size_t dof = static_cast<size_t>(std::llround(DOFmin * std::pow(log_step, istep)));

        // Match the Kokkos test span, you can easily change 10 up to 18 if you want higher degrees
        for (int nq = 3; nq <= 10; ++nq) {
            int nm = nq - 1;

            size_t nelmt = dof / (3 * nm * nm * nm);
            if (nelmt == 0) continue;

            // Template deduction handles compile-time resolution for NelmtPerBatch. 
            // 4*nq*nq*nq matches your original calculation.
            switch(nq) {
                case 3:  run_test<T, 3,  2,  std::max(1UL, shmemPerBlock / (4 * 3*3*3) / sizeof(T))>(nelmt, ntests); break;
                case 4:  run_test<T, 4,  3,  std::max(1UL, shmemPerBlock / (4 * 4*4*4) / sizeof(T))>(nelmt, ntests); break;
                case 5:  run_test<T, 5,  4,  std::max(1UL, shmemPerBlock / (4 * 5*5*5) / sizeof(T))>(nelmt, ntests); break;
                case 6:  run_test<T, 6,  5,  std::max(1UL, shmemPerBlock / (4 * 6*6*6) / sizeof(T))>(nelmt, ntests); break;
                case 7:  run_test<T, 7,  6,  std::max(1UL, shmemPerBlock / (4 * 7*7*7) / sizeof(T))>(nelmt, ntests); break;
                case 8:  run_test<T, 8,  7,  std::max(1UL, shmemPerBlock / (4 * 8*8*8) / sizeof(T))>(nelmt, ntests); break;
                case 9:  run_test<T, 9,  8,  std::max(1UL, shmemPerBlock / (4 * 9*9*9) / sizeof(T))>(nelmt, ntests); break;
                case 10: run_test<T, 10, 9,  std::max(1UL, shmemPerBlock / (4 * 10*10*10) / sizeof(T))>(nelmt, ntests); break;
                
                case 11: run_test<T, 11, 10, std::max(1UL, shmemPerBlock / (4 * 11*11*11) / sizeof(T))>(nelmt, ntests); break;
                case 12: run_test<T, 12, 11, std::max(1UL, shmemPerBlock / (4 * 12*12*12) / sizeof(T))>(nelmt, ntests); break;
                case 13: run_test<T, 13, 12, std::max(1UL, shmemPerBlock / (4 * 13*13*13) / sizeof(T))>(nelmt, ntests); break;
                case 14: run_test<T, 14, 13, std::max(1UL, shmemPerBlock / (4 * 14*14*14) / sizeof(T))>(nelmt, ntests); break;
                case 15: run_test<T, 15, 14, std::max(1UL, shmemPerBlock / (4 * 15*15*15) / sizeof(T))>(nelmt, ntests); break;
                case 16: run_test<T, 16, 15, std::max(1UL, shmemPerBlock / (4 * 16*16*16) / sizeof(T))>(nelmt, ntests); break;
                case 17: run_test<T, 17, 16, std::max(1UL, shmemPerBlock / (4 * 17*17*17) / sizeof(T))>(nelmt, ntests); break;
                case 18: run_test<T, 18, 17, std::max(1UL, shmemPerBlock / (4 * 18*18*18) / sizeof(T))>(nelmt, ntests); break;

                default: break;
            }
        }
    }

    return 0;
}