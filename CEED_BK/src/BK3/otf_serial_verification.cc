#include <iostream>
#include <cmath>
#include <algorithm>

// Make sure this points to the header where the optimized OTF kernel is saved
#include "kernels/BK3/otf_serial_kernels.hpp"

template<typename T>
void run_test(const unsigned int nq, const unsigned int nelmt){

    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;

    //Allocation of arrays
    T* basis0 = new T[nm * nq];
    T* basis1 = new T[nm * nq];
    T* basis2 = new T[nm * nq];
    T* dbasis0 = new T[nq * nq];
    T* dbasis1 = new T[nq * nq];
    T* dbasis2 = new T[nq * nq];
    T* weights = new T[nq];
    T* coord_q = new T[nelmt * 3 * nquad];
    T* in = new T[nelmt * nm * nm * nm];
    T* out = new T[nelmt * nm * nm * nm];

    //Initialize input/output and weights
    for(unsigned int i = 0; i < nelmt * nm * nm * nm; ++i)
        in[i] = std::sin((T)i);

    std::fill(out, out + nelmt * nm * nm * nm, (T)0.0);
    std::fill(weights, weights + nq, (T)1.0);

    //Initialize coord_q as the same stretched 3D grid used by the Kokkos kernel
    for(unsigned int e = 0; e < nelmt; ++e){
        unsigned int coord_base = e * 3 * nquad;
        unsigned int xbase = coord_base;
        unsigned int ybase = coord_base + nquad;
        unsigned int zbase = coord_base + 2 * nquad;

        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){
                    unsigned int idx = p * nq * nq + q * nq + r;
                    coord_q[xbase + idx] = (T)p + 0.1 * (T)q + 0.1 * (T)r;
                    coord_q[ybase + idx] = 0.1 * (T)p + (T)q + 0.1 * (T)r;
                    coord_q[zbase + idx] = 0.1 * (T)p + 0.1 * (T)q + (T)r;
                }
            }
        }
    }

    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis0[p * nm + i] = std::cos((T)(p * nm + i));
            basis1[p * nm + i] = std::cos((T)(p * nm + i));
            basis2[p * nm + i] = std::cos((T)(p * nm + i));
        }
    }

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis0[i * nq + p] = std::cos((T)(i * nq + p));
            dbasis1[i * nq + p] = std::cos((T)(i * nq + p));
            dbasis2[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    //---------------------------Serial Kernels (OTF)--------------------------------------------------------
    T SerialSumFact = BK3::Serial::SumFactorization_OTF<T>(
        nq, nq, nq, nelmt,
        basis0, basis1, basis2,
        dbasis0, dbasis1, dbasis2,
        weights, coord_q,
        in, out);

    std::cout << "SerialSumFact (OTF) norm = " << std::sqrt(std::abs(SerialSumFact)) << "\n";

    delete[] basis0; delete[] basis1; delete[] basis2;
    delete[] dbasis0; delete[] dbasis1; delete[] dbasis2;
    delete[] weights; delete[] coord_q;
    delete[] in; delete[] out;
}

int main(int argc, char **argv){
    unsigned int p = (argc > 1) ? atoi(argv[1]) : 2u;
    unsigned int nq = p + 2;
    unsigned int nelmt = (argc > 2) ? atoi(argv[2]) : 2 << 16;

    std::cout.precision(8);
    run_test<double>(nq, nelmt);

    return 0;
}