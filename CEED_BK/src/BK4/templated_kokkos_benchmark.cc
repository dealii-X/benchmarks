#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <kernels/BK4/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;
    const unsigned int ncomp = 3; // 3D Vector field has 3 components

    // Allocation of arrays
    T* basis = new T[nm * nq];
    T* dbasis = new T[nm * nq];
    T* dbasis_geo = new T[nq * nq];

    T* G = new T[nelmt * 6 * nquad];
    T* in = new T[nelmt * ncomp * nm * nm * nm];
    T* out = new T[nelmt * ncomp * nm * nm * nm];

    // Initialize the input and output arrays (3 components for vector field)
    for(unsigned int i = 0; i < nelmt * ncomp * nm * nm * nm; ++i)
        in[i] = std::sin(i); 

    std::fill(out, out + nelmt * ncomp * nm * nm * nm, (T)0.0);

    // Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[i * nq + p] = std::sin((i * nq + p));
        }
    }

    // Initialization of dbasis functions
    for(unsigned int i = 0u; i < nm; i++)
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

    // ------------------------- BK4 Vector Laplace Operator Kernel ------------------------------------
    {
        std::vector<double> results = BK4::Parallel::Kokkos_LaplaceOperator<T, nq>(
            nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
            basis, dbasis, G, in, out, ntests);

        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];
        
        uint64_t nDOF = (uint64_t)ncomp * nm * nm * nm * nelmt; 
        uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 * (6 * nDOF + 6 * nQuad) * sizeof(T) / time;
        printer("BK4", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    delete[] basis; delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    Kokkos::initialize(argc, argv);

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u; unsigned int nq = p + 2;
    size_t nelmt                   = (argc > 2) ? std::stoull(argv[2]) : (1ULL << 16);
    
    unsigned int nelmtPerBatch     = (argc > 3) ? atoi(argv[3]) : std::max(1UL, shmemPerBlock / (6 * nq * nq * nq) / sizeof(T));
    unsigned int numBlocks         = (argc > 4) ? atoi(argv[4]) : std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

    threadsPerBlock                = (argc > 5) ? atoi(argv[5]) : threadsPerBlock;
    unsigned int ntests             = (argc > 6) ? atoi(argv[6]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 10: run_test<T, 10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    Kokkos::finalize();
    return 0;
}