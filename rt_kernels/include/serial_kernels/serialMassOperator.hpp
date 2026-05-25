#ifndef SERIAL_MASS_OPERATOR_HPP
#define SERIAL_MASS_OPERATOR_HPP

#include <numeric>
#include <algorithm>

namespace Serial {

template<typename T>
T Mass(const unsigned int nq, 
    const unsigned int nm_t, const unsigned int nm_n, 
    const unsigned int nelmt,
    const T *__restrict__ basis_n, const T *__restrict__ basis_t,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out)
{
    // Total DoFs per element for a Raviart-Thomas vector
    const unsigned int ndof_1D = nm_n * nm_t * nm_t;
    const unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

    // Intermediate vals
    T *wsp0 = new T[nq * nq * nq];
    T *wsp1 = new T[nq * nq * nq];
    
    // Arrays to hold the 3 vector components at quadrature points
    T *uq_0 = new T[nq * nq * nq];
    T *uq_1 = new T[nq * nq * nq];
    T *uq_2 = new T[nq * nq * nq];


    for(unsigned int e = 0; e < nelmt; ++e){
        
        T* in_e = in + e * ndof_total;
        T* out_e = out + e * ndof_total;

        // Structure of arrays format (SoA)
        const T* in_0 = in_e;
        const T* in_1 = in_e + ndof_1D;
        const T* in_2 = in_e + 2 * ndof_1D;

        T* out_0 = out_e;
        T* out_1 = out_e + ndof_1D;
        T* out_2 = out_e + 2 * ndof_1D;
        

        // ==========================================
        // PHASE 1: Interpolate to Quadrature Nodes
        // ==========================================
        
        // --- Component 0 (x-direction) ---
        // x is normal (basis_n), y and z are tangent (basis_t)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_n; ++i)
                tmp += in_0[i*nm_t*nm_t + j*nm_t + k] * basis_n[i*nq + p];
            wsp0[p*nm_t*nm_t + j*nm_t + k] = tmp;
        }


        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = (T)0;
            for(unsigned int j=0; j<nm_t; ++j)
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * basis_t[j*nq + q];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }


        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_t; ++k)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[k*nq + r];
            uq_0[r*nq*nq + q*nq + p] = tmp;
        }
        // --- Component 1 (y-direction) ---
        // y is normal (basis_n), x and z are tangent (basis_t)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_1[i*nm_n*nm_t + j*nm_t + k] * basis_t[i*nq + p];
            wsp0[p*nm_n*nm_t + j*nm_t + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm_n; ++j)
                tmp += wsp0[p*nm_n*nm_t + j*nm_t + k] * basis_n[j*nq + q];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_t; ++k)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[k*nq + r];
            uq_1[r*nq*nq + q*nq + p] = tmp;
        }

        // --- Component 2 (z-direction) ---
        // z is normal (basis_n), x and y are tangent (basis_t)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_2[i*nm_t*nm_n + j*nm_n + k] * basis_t[i*nq + p];
            wsp0[p*nm_t*nm_n + j*nm_n + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm_t; ++j)
               tmp += wsp0[p*nm_t*nm_n + j*nm_n + k] * basis_t[j*nq + q];
            wsp1[p*nq*nm_n + q*nm_n + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_n; ++k)
               tmp += wsp1[p*nq*nm_n + q*nm_n + k] * basis_n[k*nq + r];
            uq_2[r*nq*nq + q*nq + p] = tmp;
        }


        // ==========================================
        // PHASE 2: Apply Piola Geometry Metric
        // ==========================================
        for(unsigned int r = 0; r < nq; ++r){
            for(unsigned int q = 0; q < nq; ++q){              
                for(unsigned int p = 0; p < nq; ++p){

                    int q_idx = r * nq * nq + q * nq + p;
                    int e_offset = e * 6 * nq * nq * nq;

                    // Load Piola Geometric Factors (1/|J| * J^T * J)
                    T G00 = G[e_offset + 0 * nq*nq*nq + q_idx];
                    T G01 = G[e_offset + 1 * nq*nq*nq + q_idx];
                    T G02 = G[e_offset + 2 * nq*nq*nq + q_idx];
                    T G11 = G[e_offset + 3 * nq*nq*nq + q_idx];
                    T G12 = G[e_offset + 4 * nq*nq*nq + q_idx];
                    T G22 = G[e_offset + 5 * nq*nq*nq + q_idx];
                    
                    T u0 = uq_0[q_idx];
                    T u1 = uq_1[q_idx];
                    T u2 = uq_2[q_idx];

                    // Coupled Matrix-Vector multiply
                    uq_0[q_idx] = G00 * u0 + G01 * u1 + G02 * u2;
                    uq_1[q_idx] = G01 * u0 + G11 * u1 + G12 * u2;
                    uq_2[q_idx] = G02 * u0 + G12 * u1 + G22 * u2;
                }
            }
        }


        // ==========================================
        // PHASE 3: Project back to Nodes (Transpose)
        // ==========================================
        
        // --- Component 0 (x-direction) ---
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_0[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[j*nq + q];
            wsp0[p*nm_t*nm_t + j*nm_t + k] = tmp;
        }
        for(unsigned int i=0; i<nm_n; ++i)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * basis_n[i*nq + p];
            out_0[i*nm_t*nm_t + j*nm_t + k] = tmp;
        }

        // --- Component 1 (y-direction) ---
        // Normal in y (basis_n), tangent in x and z (basis_t)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_1[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_n[j*nq + q];
                wsp0[p*nm_n*nm_t + j*nm_t + k] = tmp;
                
        }

        for(unsigned int i=0; i<nm_t; ++i)
        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_n*nm_t + j*nm_t + k] * basis_t[i*nq + p];
            out_1[i*nm_n*nm_t + j*nm_t + k] = tmp;
        }

        // --- Component 2 (z-direction) ---
        // Normal in z (basis_n), tangent in x and y (basis_t)
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_2[r*nq*nq + q*nq + p] * basis_n[k*nq + r];
            wsp1[p*nq*nm_n + q*nm_n + k] = tmp;
        }

        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_n + q*nm_n + k] * basis_t[j*nq + q];
            wsp0[p*nm_t*nm_n + j*nm_n + k] = tmp;
        }

        for(unsigned int i=0; i<nm_t; ++i)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_t*nm_n + j*nm_n + k] * basis_t[i*nq + p];
            out_2[i*nm_t*nm_n + j*nm_n + k] = tmp;
        }

    }

    delete[] wsp0; delete[] wsp1; delete[] uq_0; delete[] uq_1; delete[] uq_2;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
#endif   //SERIAL_MASS_OPERATOR_HPP