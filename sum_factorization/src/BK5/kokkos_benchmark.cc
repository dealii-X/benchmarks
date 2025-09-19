#include <iostream>
#include <kernels/BK5/kokkos_kernels.hpp>
#include <timer.hpp>
#include <iomanip>

template<typename T>
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int numThreads3D, const unsigned int nelmt, const unsigned int ntests)
{
    //Allocation of arrays
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

    std::cout << std::fixed << std::setprecision(3);

    std::cout << std::left  << std::setw(15) << "Kernel"
              << std::right << std::setw(4)  << "p0"
              << std::right << std::setw(4)  << "p1"
              << std::right << std::setw(4)  << "p2"
              << std::right << std::setw(12)  << "nelmt"
              << std::right << std::setw(16) << "numThreads"
              << std::right << std::setw(16)  << "DOF"
              << std::right << std::setw(10)  << "time"
              << std::right << std::setw(8)  << "GDOF/s"
              << std::endl;

    // ------------------------- 1D Block Simple Map Kernel ---------------------------------------------------
    {
        std::vector<T> results = Parallel::KokkosKernel_3D_Block_SimpleMap<T>(nq0 ,nq1, nq2, dbasis0, dbasis1, dbasis2, G, in, out, numThreads3D, nelmt, ntests);
        std::cout << std::left  << std::setw(15) << "1DS" 
                  << std::right << std::setw(4)  << nq0 - 1 
                  << std::right << std::setw(4)  << nq1 - 1
                  << std::right << std::setw(4)  << nq2 - 1
                  << std::right << std::setw(12) << nelmt
                  << std::right << std::setw(16) << numThreads3D
                  << std::right << std::setw(16) << nq0 * nq1 * nq2 * nelmt
                  << std::right << std::setw(10) << results[2]
                  << std::right << std::setw(8)  << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / results[2]
                  << std::endl;
    }

    // ------------------------- 2D Block(jk) Simple Map Kernel ---------------------------------------------------
    {
        std::vector<T> results = Parallel::KokkosKernel_2D_Block_jk_SimpleMap<T>(nq0 ,nq1, nq2, dbasis0, dbasis1, dbasis2, G, in, out, numThreads3D, nelmt, ntests);
        std::cout << std::left  << std::setw(15) << "2DS(jk)" 
                  << std::right << std::setw(4)  << nq0 - 1 
                  << std::right << std::setw(4)  << nq1 - 1
                  << std::right << std::setw(4)  << nq2 - 1
                  << std::right << std::setw(12) << nelmt
                  << std::right << std::setw(16) << numThreads3D / nq0
                  << std::right << std::setw(16) << nq0 * nq1 * nq2 * nelmt
                  << std::right << std::setw(10) << results[2]
                  << std::right << std::setw(8)  << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / results[2]
                  << std::endl;
    }

    
    delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){
    unsigned int nq0               = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1               = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2               = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt             = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    unsigned int numThreads3D      = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int ntests            = (argc > 6) ? atoi(argv[6]) : 50u;

    Kokkos::initialize(argc, argv);

    std::cout.precision(8);
    run_test<float>(nq0, nq1, nq2, numThreads3D, nelmt, ntests);

    Kokkos::finalize();
    return 0;
}