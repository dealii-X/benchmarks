#include <iostream>
#include <cmath>
#include <algorithm> 
#include "serialMassOperator.hpp"
#include "serialMixedDivOperator.hpp" 
#include "serialMixedGradOperator.hpp" 
#include "serialDivDivOperator.hpp" 


template<typename T>
void run_test(const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nelmt){

    // 1. DoFs for an anisotropic Raviart-Thomas vector (Velocity space)
    const unsigned int ndof_x = nm_n * nm_t * nm_t;
    const unsigned int ndof_y = nm_t * nm_n * nm_t;
    const unsigned int ndof_z = nm_t * nm_t * nm_n;
    const unsigned int ndof_u_total = ndof_x + ndof_y + ndof_z;

    // 2. DoFs for Scalar space (Pressure space)
    const unsigned int nm_p = nm_t; 
    const unsigned int ndof_p_total = nm_p * nm_p * nm_p;

    T* basis_n  = new T[nq * nm_n];
    T* basis_t  = new T[nq * nm_t];
    T* dbasis_n = new T[nq * nm_n];
    T* basis_p  = new T[nq * nm_p];

    // Separate metrics
    T* G_mass = new T[nelmt * 6 * nq * nq * nq];
    T* G_div  = new T[nelmt * 1 * nq * nq * nq];
    
    // Arrays for Velocity (u) and Pressure (p)
    T* in_u  = new T[nelmt * ndof_u_total];
    T* out_u = new T[nelmt * ndof_u_total];
    T* in_p  = new T[nelmt * ndof_p_total];
    T* out_p = new T[nelmt * ndof_p_total];

    // Initialization
    std::fill(in_u, in_u + nelmt * ndof_u_total, (T)3.0);
    std::fill(in_p, in_p + nelmt * ndof_p_total, (T)3.0);
    

    // Metric Tensor G_mass
    for(unsigned int i = 0u; i < nelmt * 6 * nq * nq * nq; i++) {
            G_mass[i] = std::cos(i);
    }

    // Metric Tensor div
    for(unsigned int i = 0u; i < nelmt * 1 * nq * nq * nq; i++) {
            G_div[i] = std::cos(i);
    }


    // Normal basis (nm_n x nq)
    for(unsigned int i = 0u; i < nm_n; i++) {
        for(unsigned int p = 0u; p < nq; p++) {
            basis_n[i * nq + p] = std::cos((T)(i * nq + p));
            dbasis_n[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }
    
    // Tangent basis (nm_t x nq)
    for(unsigned int j = 0u; j < nm_t; j++) {
        for(unsigned int q = 0u; q < nq; q++) {
            basis_t[j * nq + q] = std::cos((T)(j * nq + q));
        }
    }

    // Pressure basis (nm_p x nq)
    for(unsigned int k = 0u; k < nm_p; k++) {
        for(unsigned int r = 0u; r < nq; r++) {
            basis_p[k * nq + r] = std::cos((T)(k * nq + r)); 
        }
    }

    // ==========================================
    // 1. Calculate HDivMass (u -> u)
    // ==========================================
    std::fill(out_u, out_u + nelmt * ndof_u_total, (T)0.0);

    T Serial_Mass = Serial::Mass<T>(
        nq, nm_t, nm_n, nelmt, 
        basis_n, basis_t, 
        G_mass, in_u, out_u
    );
    std::cout << "Serial_Mass L2 norm = " << std::sqrt(Serial_Mass) << "\n";


    // ==========================================
    // 2. Calculate MixedDiv (u -> p)
    // ==========================================
    std::fill(out_p, out_p + nelmt * ndof_p_total, (T)0.0);

    T Serial_MixedDiv = Serial::MixedDiv<T>(
        nq, nm_t, nm_n, nm_p, nelmt, 
        dbasis_n, basis_t, basis_p,
        G_div, in_u, out_p // Velocity IN, Pressure OUT
    );
    std::cout << "Serial_MixedDiv L2 norm = " << std::sqrt(Serial_MixedDiv) << "\n";


    // ==========================================
    // 3. Calculate MixedGrad (p -> u)
    // ==========================================
    std::fill(out_u, out_u + nelmt * ndof_u_total, (T)0.0);

    T Serial_MixedGrad = Serial::MixedGrad<T>(
        nq, nm_t, nm_n, nm_p, nelmt, 
        dbasis_n, basis_t, basis_p,
        G_div, in_p, out_u // Pressure IN, Velocity OUT
    );
    std::cout << "Serial_MixedGrad L2 norm = " << std::sqrt(Serial_MixedGrad) << "\n";


    // ==========================================
    // 4. Calculate DivDiv (u -> u)
    // ==========================================
    std::fill(out_u, out_u + nelmt * ndof_u_total, (T)0.0);

    T Serial_DivDiv_res = Serial::DivDiv<T>(
        nq, nm_t, nm_n, nelmt, 
        dbasis_n, basis_t, 
        G_div, in_u, out_u
    );
    std::cout << "Serial_DivDiv L2 norm = " << std::sqrt(Serial_DivDiv_res) << "\n";

    delete[] basis_n;  delete[] basis_t;  delete[] dbasis_n; delete[] basis_p; delete[] G_mass;  delete[] G_div; delete[] in_u; delete[] out_u; delete[] in_p; delete[] out_p;
}

int main(int argc, char **argv){

    unsigned int p = (argc > 1) ? std::atoi(argv[1]) : 2u; 
    unsigned int nelmt = (argc > 2) ? std::atoi(argv[2]) : 2 << 15;

    unsigned int nm_t = p + 1;
    unsigned int nm_n = p + 2;
    unsigned int nq = p + 2; 
        
    std::cout.precision(8);
    
    run_test<double>(nq, nm_t, nm_n, nelmt);
    
    return 0;
}