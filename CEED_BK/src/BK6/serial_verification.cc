#include <iostream>
#include <cmath>
#include <algorithm>
#include <kernels/BK6/serial_kernels.hpp>

template<typename T>
void run_test(const unsigned int nq, const unsigned int nelmt){

    const unsigned int nquad = nq * nq * nq;

    // Total DoFs per element for a 3-component vector field
    const unsigned int ndof_1D = nq * nq * nq;
    const unsigned int ndof_total = 3 * ndof_1D;

    //Allocation of arrays
    T* dbasis = new T[nq * nq];

    T* G = new T[nelmt * 6 * nquad];
    T* in = new T[nelmt * ndof_total];
    T* out = new T[nelmt * ndof_total];

    //Initialize the input and output arrays
    for(unsigned int i = 0; i < nelmt * ndof_total; ++i)
        in[i] = std::sin(i);

    std::fill(out, out + nelmt * ndof_total, (T)0.0);


    //Initialization of derivative basis functions
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




    //---------------------------Serial Kernels--------------------------------------------------------
    T SerialSumFact = BK6::Serial::SumFactorization<T>(
        nq, nelmt, dbasis,
        G, in, out);

    std::cout << "SerialSumFact norm = " << std::sqrt(std::abs(SerialSumFact)) << "\n";

    delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){
    unsigned int p     = (argc > 1) ? atoi(argv[1]) : 2u;
    unsigned int nq    = p + 1;
    unsigned int nelmt = (argc > 2) ? atoi(argv[2]) : 1 << 16;

    std::cout.precision(8);
    run_test<double>(nq, nelmt);

    return 0;
}