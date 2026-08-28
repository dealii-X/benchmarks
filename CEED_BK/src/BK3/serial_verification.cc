#include <iostream>
#include <cmath>
#include <algorithm>
#include <kernels/BK3/serial_kernels.hpp>

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
    T* G = new T[nelmt * 6 * nquad];
    T* in = new T[nelmt * nm * nm * nm];
    T* out = new T[nelmt * nm * nm * nm];

    //Initialize the input and output arrays
    for(unsigned int i = 0; i < nelmt * nm * nm * nm; ++i)
        in[i] = std::sin((T)i);

    std::fill(out, out + nelmt * nm * nm * nm, (T)0.0);
    std::fill(weights, weights + nq, (T)1.0);

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

    //Compute G from coord_q using the same J -> C -> detJ formulation as OTF
    for(unsigned int e = 0; e < nelmt; ++e){
        unsigned int coord_base = e * 3 * nquad;
        unsigned int xbase = coord_base;
        unsigned int ybase = coord_base + nquad;
        unsigned int zbase = coord_base + 2 * nquad;
        unsigned int Gbase = e * 6 * nquad;

        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){

                    T J00 = 0.0, J01 = 0.0, J02 = 0.0;
                    T J10 = 0.0, J11 = 0.0, J12 = 0.0;
                    T J20 = 0.0, J21 = 0.0, J22 = 0.0;

                    for(unsigned int n = 0; n < nq; ++n){
                        const T x_r = coord_q[xbase + n * nq * nq + q * nq + r];
                        const T x_s = coord_q[xbase + p * nq * nq + n * nq + r];
                        const T x_t = coord_q[xbase + p * nq * nq + q * nq + n];

                        J00 += dbasis0[n * nq + p] * x_r;
                        J01 += dbasis1[n * nq + q] * x_s;
                        J02 += dbasis2[n * nq + r] * x_t;

                        const T y_r = coord_q[ybase + n * nq * nq + q * nq + r];
                        const T y_s = coord_q[ybase + p * nq * nq + n * nq + r];
                        const T y_t = coord_q[ybase + p * nq * nq + q * nq + n];

                        J10 += dbasis0[n * nq + p] * y_r;
                        J11 += dbasis1[n * nq + q] * y_s;
                        J12 += dbasis2[n * nq + r] * y_t;

                        const T z_r = coord_q[zbase + n * nq * nq + q * nq + r];
                        const T z_s = coord_q[zbase + p * nq * nq + n * nq + r];
                        const T z_t = coord_q[zbase + p * nq * nq + q * nq + n];

                        J20 += dbasis0[n * nq + p] * z_r;
                        J21 += dbasis1[n * nq + q] * z_s;
                        J22 += dbasis2[n * nq + r] * z_t;
                    }

                    const T C00 = J11 * J22 - J12 * J21;
                    const T C01 = J02 * J21 - J01 * J22;
                    const T C02 = J01 * J12 - J02 * J11;

                    const T C10 = J12 * J20 - J10 * J22;
                    const T C11 = J00 * J22 - J02 * J20;
                    const T C12 = J02 * J10 - J00 * J12;

                    const T C20 = J10 * J21 - J11 * J20;
                    const T C21 = J01 * J20 - J00 * J21;
                    const T C22 = J00 * J11 - J01 * J10;

                    const T detJ = J00 * C00 + J01 * C10 + J02 * C20;
                    const T scale = (weights[p] * weights[q] * weights[r]) / detJ;

                    const T G00 = scale * (C00 * C00 + C01 * C01 + C02 * C02);
                    const T G01 = scale * (C00 * C10 + C01 * C11 + C02 * C12);
                    const T G02 = scale * (C00 * C20 + C01 * C21 + C02 * C22);
                    const T G11 = scale * (C10 * C10 + C11 * C11 + C12 * C12);
                    const T G12 = scale * (C10 * C20 + C11 * C21 + C12 * C22);
                    const T G22 = scale * (C20 * C20 + C21 * C21 + C22 * C22);

                    unsigned int idx = p * nq * nq + q * nq + r;

                    G[Gbase + 0 * nquad + idx] = G00;
                    G[Gbase + 1 * nquad + idx] = G01;
                    G[Gbase + 2 * nquad + idx] = G02;
                    G[Gbase + 3 * nquad + idx] = G11;
                    G[Gbase + 4 * nquad + idx] = G12;
                    G[Gbase + 5 * nquad + idx] = G22;
                }
            }
        }
    }

    //---------------------------Serial Kernels--------------------------------------------------------
    T SerialSumFact = BK3::Serial::SumFactorization<T>(
        nq, nq, nq, nelmt,
        basis0, basis1, basis2,
        dbasis0, dbasis1, dbasis2,
        G, in, out);

    std::cout << "SerialSumFact norm = " << std::sqrt(std::abs(SerialSumFact)) << "\n";

    delete[] basis0; delete[] basis1; delete[] basis2; delete[] dbasis0; delete[] dbasis1; delete[] dbasis2;
    delete[] weights; delete[] coord_q; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){
    unsigned int p     = (argc > 1) ? atoi(argv[1]) : 2u;
    unsigned int nq    = p + 2;
    unsigned int nelmt = (argc > 2) ? atoi(argv[2]) : 2 << 16;

    std::cout.precision(8);
    run_test<double>(nq, nelmt);

    return 0;
}