/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <iostream>
#include <kernels/BK1/templated_kokkos_kernels.hpp>
#include <timer.hpp>

template<typename T, unsigned int nq0, unsigned int nq1, unsigned int nq2>
void run_test(
    const unsigned int numThreads, const unsigned int threadsPerBlock, 
    const unsigned int nelmt, const unsigned int ntests)
{
    const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;
    
    //Allocation of arrays
    T* basis0 = new T[nm0 * nq0];
    T* basis1 = new T[nm1 * nq1];
    T* basis2 = new T[nm2 * nq2];
    T* JxW = new T[nelmt * nq0 * nq1 * nq2];
    T* in = new T[nelmt * nm0 * nm1 * nm2];
    T* out = new T[nelmt * nm0 * nm1 * nm2];


    //Initialize the input and output arrays
    std::fill(JxW, JxW + nelmt * nq0 * nq1 * nq2, (T)1.0f);
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (T)3.0f);
    std::fill(out, out + nelmt * nm0 * nm1 * nm2, (T)0.0f);


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

    // ------------------------- Kokkos Kernel ---------------------------------------------------
    std::vector<T> results = Parallel::KokkosKernel<T,nq0 ,nq1, nq2>(basis0, basis1, basis2, JxW, in, out, numThreads, threadsPerBlock, nelmt, ntests);
    std::cout << "Kokkos -> " << "nelmt = " << nelmt <<" GDoF/s = " << results[0] << std::endl;
    

    delete[] basis0; delete[] basis1; delete[] basis2; delete[] JxW; delete[] in; delete[] out;
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