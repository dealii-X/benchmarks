#include <iostream>
#include <kernels/BK3/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
void run_test( const unsigned int numThreads, const unsigned int threadsPerBlock, const unsigned int nelmt, const unsigned int ntests)
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
    T* G = new T[nelmt * nq0 * nq1 * nq2 * 6];
    T* in = new T[nelmt * nq0 * nq1 * nq2];
    T* out = new T[nelmt * nq0 * nq1 * nq2];


    //Initialize the input and output arrays
    std::fill(G, G + nelmt * 6 * nq0 * nq1 * nq2, (T)2.0f);
    std::fill(in, in + nelmt * nq0 * nq1 * nq2, (T)3.0f);
    std::fill(out, out + nelmt * nq0 * nq1 * nq2, (T)0.0f);


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

    BenchmarkPrinter printer;
    printer.print_header();

    // ------------------------- 1D Block Kernel ---------------------------------------------------
    {
        std::vector<T> results = BK3::Parallel::KokkosKernel_1D_Block<T, nq0 ,nq1, nq2>(basis0, basis1, basis2, 
                                            dbasis0, dbasis1, dbasis2, G, in, out,
                                            numThreads, threadsPerBlock, nelmt, ntests);

        printer("1D", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numThreads, nm0 * nm1 * nm2 * nelmt, results[2], 1.0e-9 * nelmt * nm0 * nm1 * nm2 / results[2]);
    }

    // ------------------------- 1D Block Kernel + Simple Map ---------------------------------------------
    {
        std::vector<T> results = BK3::Parallel::KokkosKernel_1D_Block_SimpleMap<T, nq0 ,nq1, nq2>(basis0, basis1, basis2, 
                                            dbasis0, dbasis1, dbasis2, G, in, out, numThreads, nelmt, ntests);
        
        printer("1DS", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numThreads, nm0 * nm1 * nm2 * nelmt, results[2], 1.0e-9 * nelmt * nm0 * nm1 * nm2 / results[2]);
    }

    // ------------------------- 2D Block(pq) Kernel ---------------------------------------------------
    {
        std::vector<T> results = BK3::Parallel::KokkosKernel_2D_Block_pq<T, nq0 ,nq1, nq2>(basis0, basis1, basis2, 
                                            dbasis0, dbasis1, dbasis2, G, in, out,
                                            numThreads, threadsPerBlock, nelmt, ntests);

        printer("2D", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numThreads / nq2, nm0 * nm1 * nm2 * nelmt, results[2], 1.0e-9 * nelmt * nm0 * nm1 * nm2 / results[2]);
    }

    // ------------------------- 2D Block(pq) Simple Map Kernel ---------------------------------------------------
    {
        std::vector<T> results = BK3::Parallel::KokkosKernel_2D_Block_pq_SimpleMap<T, nq0 ,nq1, nq2>(basis0, basis1, basis2, 
                                            dbasis0, dbasis1, dbasis2, G, in, out,
                                            numThreads, nelmt, ntests);

        printer("2DS", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numThreads / nq2, nm0 * nm1 * nm2 * nelmt, results[2], 1.0e-9 * nelmt * nm0 * nm1 * nm2 / results[2]);
    }


    delete[] basis0; delete[] basis1; delete[] basis2; delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){
    constexpr unsigned int nq0      = 4u;
    constexpr unsigned int nq1      = 4u;
    constexpr unsigned int nq2      = 4u;

    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    unsigned int numThreads         = (argc > 2) ? atoi(argv[2]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int threadsPerBlock    = (argc > 3) ? atoi(argv[3]) : nq0 * nq1 * nq2;
    unsigned int ntests             = (argc > 4) ? atoi(argv[4]) : 50u;

    Kokkos::initialize(argc, argv);

    std::cout.precision(8);
    run_test<float, nq0, nq1, nq2>(numThreads, threadsPerBlock, nelmt, ntests);

    Kokkos::finalize();
    return 0;
}