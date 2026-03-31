#include <iostream>
#include <kernels/BK3/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;

    //Allocation of arrays
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];

    T* G = new T[nelmt * 6 * nq*nq*nq];
    T* in = new T[nelmt * nm*nm*nm];
    T* out = new T[nelmt * nm*nm*nm];


    //Initialize the input and output arrays
    std::fill(G,   G + nelmt * nq*nq*nq * 6, (T)2.0f);
    std::fill(in,  in + nelmt * nm*nm*nm, (T)3.0f);
    std::fill(out, out + nelmt * nm*nm*nm, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[p * nm + i] = std::cos((T)(p * nm + i));
        }
    }

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    BenchmarkPrinter<T> printer;
    printer.print_header();



    // ------------------------- 2D Block(pq) Kernel ---------------------------------------------------
    {
        std::vector<double> results = BK3::Parallel::Kokkos_LaplaceOperator<T, nq>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
                                            basis, dbasis, G, in, out, ntests);
        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];
        
        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt; uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 *(2 * nDOF + 6*nQuad) * sizeof(T) / time;
        printer("BK3", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }


    delete[] basis; delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = float;
    int shmemPerBlock = 10800;

    Kokkos::initialize(argc, argv);

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u; unsigned int nq = p + 2;
    unsigned int nelmt             = (argc > 2) ? atoi(argv[2]) : 2 << 18;
    
    unsigned int nelmtPerBatch     = shmemPerBlock / (4 * nq * nq * nq) / sizeof(T);    if(nelmtPerBatch == 0) nelmtPerBatch = 1;
    unsigned int numBlocks         = (argc > 3) ? atoi(argv[3]) : (nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2;

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

    threadsPerBlock                = (argc > 4) ? atoi(argv[4]) : threadsPerBlock;
    unsigned int ntests            = (argc > 5) ? atoi(argv[5]) : 10u;

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