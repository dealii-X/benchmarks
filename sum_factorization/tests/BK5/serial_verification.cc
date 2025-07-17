#include <iostream>
#include <cmath>
#include <kernels/BK5/serial_kernels.hpp>

template<typename T>
void run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt){

    //Allocation of arrays
    T* dbasis0 = new T[nq0 * nq0];
    T* dbasis1 = new T[nq1 * nq1];
    T* dbasis2 = new T[nq2 * nq2];
    T* G = new T[nelmt * nq0 * nq1 * 6 * nq2];
    T* in = new T[nelmt * nq0 * nq1 * nq2];
    T* out = new T[nelmt * nq0 * nq1 * nq2];

    //Initialize the input and output arrays
    std::fill(G, G + nelmt * nq0 * nq1 * nq2 * 6, (T)2.0f);
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

    //---------------------------Serial Kernels--------------------------------------------------------
    T SerialSumFact = BK5::Serial::SumFactorization<T>(nq0, nq1, nq2, nelmt, dbasis0, dbasis1, dbasis2, G, in, out);

    std::cout << "SerialSumFact norm = " << std::sqrt(SerialSumFact) << "\n";

    delete[] dbasis0; delete[] dbasis1; delete[] dbasis2; delete[] G; delete[] in;
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