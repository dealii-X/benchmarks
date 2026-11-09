#include <iostream>
#include <cmath>
#include <algorithm>
#include <kernels/BK2/serial_kernels.hpp>

template<typename T>
void run_test(const unsigned int nq, const unsigned int nelmt){
    const unsigned int nm = nq - 1;

    const unsigned int ncomp = 3;

    //Allocation of arrays
    T* basis = new T[nm * nq];

    T* JxW = new T[nelmt * nq * nq * nq];
    T* in  = new T[nelmt * ncomp * nm * nm * nm];
    T* out = new T[nelmt * ncomp * nm * nm * nm];

    //Initialize the input and output arrays
    std::fill(JxW, JxW + nelmt * nq * nq * nq, T(1.0));
    std::fill(in, in + nelmt * ncomp * nm * nm * nm, T(3.0));
    std::fill(out, out + nelmt * ncomp * nm * nm * nm, T(0.0));

    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }


    //---------------------------Serial Kernels--------------------------------------------------------
    T SerialSumFact = BK2::Serial::SumFactorization<T>(nq, nelmt, basis, JxW, in, out);

    std::cout << "Serial norm = " << std::sqrt(SerialSumFact) << "\n";

    delete[] basis; delete[] JxW; delete[] in; delete[] out;
}


int main(int argc, char **argv){
    unsigned int p                  = (argc > 1) ? atoi(argv[1]) : 2u;  unsigned int nq = p + 2;
    unsigned int nelmt              = (argc > 2) ? atoi(argv[2]) : 1 << 16;


    std::cout.precision(8);
    run_test<double>(nq, nelmt);

    return 0;
}