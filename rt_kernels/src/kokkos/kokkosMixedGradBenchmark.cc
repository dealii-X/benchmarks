#include <iostream>
#include <cmath>
#include <vector>
#include <Kokkos_Core.hpp>

#include "kokkosMixedGradOperator.hpp" // Changed header
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nm_p>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
              const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    constexpr unsigned int ndof_u_1D = nm_n * nm_t * nm_t;
    constexpr unsigned int ndof_u_total = ndof_u_1D * 3;

    constexpr unsigned int ndof_p_total = nm_p * nm_p * nm_p;

    T* dbasis_n = new T[nm_n * nq];
    T* basis_t  = new T[nm_t * nq];
    T* basis_p  = new T[nm_p * nq];

    T* G_scalar = new T[nq * nq * nq];
    
    T* in_p     = new T[nelmt * ndof_p_total];
    T* out_u    = new T[nelmt * ndof_u_total];

    std::fill(in_p,  in_p + nelmt * ndof_p_total, (T)3.0f);
    std::fill(out_u, out_u + nelmt * ndof_u_total, (T)0.0f);

    
    for(unsigned int i = 0u; i < nq * nq * nq; i++) {
        G_scalar[i] = std::cos((T)i);
    }

    for(unsigned int i = 0u; i < nm_n; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            dbasis_n[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    for(unsigned int i = 0u; i < nm_t; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis_t[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    for(unsigned int i = 0u; i < nm_p; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis_p[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- MixedGrad Operator Kernel ----------------------------------------------
    {
        std::vector<double> results = Parallel::Kokkos_MixedGrad<T, nq, nm_t, nm_n, nm_p>(
            nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
            dbasis_n, basis_t, basis_p, G_scalar, in_p, out_u, ntests
        );
        
        auto DOFs = results[0]; 
        auto sum  = results[1]; 
        auto time = results[2];
        
        uint64_t nDOF_U = (uint64_t)ndof_u_total * nelmt; 
        uint64_t nDOF_P = (uint64_t)ndof_p_total * nelmt; 
        
        T bw = 1.0e-9 * (nDOF_P + nDOF_U) * sizeof(T) / time; 
        
        printer("MixedGrad", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF_P, time, DOFs, bw, std::sqrt(sum));
    }

    delete[] dbasis_n; delete[] basis_t; delete[] basis_p; delete[] G_scalar; delete[] in_p; delete[] out_u;
}

int main(int argc, char **argv){

    using T = double;
    
    Kokkos::initialize(argc, argv);

    size_t shmemPerBlock = Kokkos::TeamPolicy<>::scratch_size_max(0);    //maximum shared memory size per thread block

    unsigned int p      = (argc > 1) ? atoi(argv[1]) : 2u;    
    unsigned int nq     = p + 2;
    unsigned int nelmt  = (argc > 2) ? atoi(argv[2]) : 2 << 15;
    
    unsigned int nelmtPerBatch = std::max(1UL, shmemPerBlock / (5 * nq * nq * nq) / sizeof(T));
    unsigned int numBlocks     = (argc > 3) ? atoi(argv[3]) : std::max(1U, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);

    unsigned int threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);
    threadsPerBlock              = (argc > 4) ? atoi(argv[4]) : threadsPerBlock;
    threadsPerBlock              = std::min(threadsPerBlock, 512u);

    
    unsigned int ntests          = (argc > 5) ? atoi(argv[5]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        // Syntax: run_test<T, nq, nm_t, nm_n, nm_p>(...)
        case 3:  run_test<T, 3,  2, 3, 2>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4:  run_test<T, 4,  3, 4, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5:  run_test<T, 5,  4, 5, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6:  run_test<T, 6,  5, 6, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7:  run_test<T, 7,  6, 7, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8:  run_test<T, 8,  7, 8, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9:  run_test<T, 9,  8, 9, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 10: run_test<T, 10, 9, 10, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    Kokkos::finalize();
    return 0;
}