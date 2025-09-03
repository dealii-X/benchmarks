#include <iostream>
#include <cmath>
#include <kernels/BK3/serial_kernels.hpp>

template<typename T>
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt){

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

    T* G = new T[nelmt * nq0 * nq1 * 6 * nq2];
    T* in = new T[nelmt * nm0 * nm1 * nm2];
    T* out = new T[nelmt * nm0 * nm1 * nm2];

    //Initialize the input and output arrays
    std::fill(G, G + nelmt * nq0 * nq1 * nq2 * 6, (T)2.0f);
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

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq0; i++)
    {
        for(unsigned int p = 0u; p < nq0; p++)
        {
            dbasis0[i * nq0 + p] = std::cos((T)(i * nq0 + p));
        }
    }
    for(unsigned int j = 0u; j < nq1; j++)
    {
        for(unsigned int q = 0u; q < nq1; q++)
        {
            dbasis1[j * nq1 + q] = std::cos((T)(j * nq1 + q));
        }
    }
    for(unsigned int k = 0u; k < nq2; k++)
    {
        for(unsigned int r = 0u; r < nq2; r++)
        {
            dbasis2[k * nq2 + r] = std::cos((T)(k * nq2 + r));
        }
    }

    //---------------------------Serial Kernels--------------------------------------------------------
    T SerialSumFact = BK3::Serial::SumFactorization<T>(nq0, nq1, nq2, nelmt, basis0, basis1, basis2, dbasis0, dbasis1, dbasis2, G, in, out);

    std::cout << "SerialSumFact norm = " << std::sqrt(SerialSumFact) << "\n";

    delete[] basis0; delete[] basis1; delete[] basis2; delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in;
}


int main(int argc, char **argv){
    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 17;
        
    std::cout.precision(8);
    run_test<double>(nq0, nq1, nq2, nelmt);
    
    return 0;
}