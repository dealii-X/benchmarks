#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <kernels/BK6/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nquad = nq * nq * nq;
    const unsigned int ncomp = 3; // 3D Vector field has 3 components

    // Allocation of arrays
    T* dbasis = new T[nq * nq];

    T* G = new T[nelmt * 6 * nquad];
    T* in = new T[nelmt * ncomp * nq * nq * nq];
    T* out = new T[nelmt * ncomp * nq * nq * nq];

    // Initialize the input and output arrays (3 components for vector field)
    for(unsigned int i = 0; i < nelmt * ncomp * nq * nq * nq; ++i)
        in[i] = std::sin(i); 

    std::fill(out, out + nelmt * ncomp * nq * nq * nq, (T)0.0);

    // Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis[i * nq + p] = std::cos((i * nq + p));
        }
    }

    //initialize G
    for(unsigned int i = 0u; i < nelmt * 6 * nquad; i++)
    {
        G[i] = std::cos((i));
    }
    
    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- BK6 Vector Laplace Operator Kernel ------------------------------------
    {
        std::vector<double> results = BK6::Parallel::Kokkos_LaplaceOperator<T, nq>(
            nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
            dbasis, G, in, out, ntests);

        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];
        
        uint64_t nDOF = (uint64_t)ncomp * nq * nq * nq * nelmt; 
        uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 * (2 * nDOF + 6 * nQuad) * sizeof(T) / time;
        printer("BK6", nq - 1, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    Kokkos::initialize(argc, argv);

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u; unsigned int nq = p + 1;
    size_t nelmt                   = (argc > 2) ? std::stoull(argv[2]) : (1ULL << 16);
    
    unsigned int nelmtPerBatch     = (argc > 3) ? atoi(argv[3]) : std::max(1UL, shmemPerBlock / (4 * nq * nq * nq) / sizeof(T));
    unsigned int numBlocks         = (argc > 4) ? atoi(argv[4]) : std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

    threadsPerBlock                = (argc > 5) ? atoi(argv[5]) : threadsPerBlock;
    unsigned int ntests             = (argc > 6) ? atoi(argv[6]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 2: run_test<T, 2>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;

        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    Kokkos::finalize();
    return 0;
}