#ifndef SERIAL_DIVDIV_OPERATOR_HPP
#define SERIAL_DIVDIV_OPERATOR_HPP

#include <numeric>
#include <algorithm>

namespace Serial {

template<typename T>
T DivDiv(const unsigned int nq, 
    const unsigned int nm_t, const unsigned int nm_n, 
    const unsigned int nelmt,
    const T *__restrict__ dbasis_n, const T *__restrict__ basis_t,
    const T *__restrict__ G_scalar, T *__restrict__ in, T * __restrict__ out)
{
    // Total DoFs per element for a Raviart-Thomas vector
    const unsigned int ndof_1D = nm_n * nm_t * nm_t;
    const unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

  
    T *wsp0  = new T[nq * nq * nq];
    T *wsp1  = new T[nq * nq * nq];
    T *accum = new T[nq * nq * nq]; // Holds the total scalar divergence

    for(unsigned int e = 0; e < nelmt; ++e){
        
        T* in_e = in + e * ndof_total;
        T* out_e = out + e * ndof_total;
        
        T* in_0 = in_e;
        T* in_1 = in_e + ndof_1D;
        T* in_2 = in_e + ndof_1D + ndof_1D;

        T* out_0 = out_e;
        T* out_1 = out_e + ndof_1D;
        T* out_2 = out_e + ndof_1D + ndof_1D;

        std::fill(accum, accum + nq * nq * nq, (T)0);

        // ==========================================
        // PHASE 1: Compute and Accumulate Reference Divergence
        // ==========================================
        
        // --- Component 0 (x-direction derivative) ---
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_n; ++i)
                tmp += in_0[i*nm_t*nm_t + j*nm_t + k] * dbasis_n[i*nq + p];
            wsp0[p*nm_t*nm_t + j*nm_t + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm_t; ++j)
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * basis_t[j*nq + q];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        // Add directly into accum
        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_t; ++k)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[k*nq + r];
            accum[p*nq*nq + q*nq + r] = tmp;
        }

        // --- Component 1 (y-direction derivative) ---
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
                tmp += wsp0[p*nm_n*nm_t + j*nm_t + k] * dbasis_n[j*nq + q];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        // Add directly into accum
        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_t; ++k)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[k*nq + r];
            accum[p*nq*nq + q*nq + r] += tmp;
        }

        // --- Component 2 (z-direction derivative) ---
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

        // Add directly into accum
        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_n; ++k)
                tmp += wsp1[p*nq*nm_n + q*nm_n + k] * dbasis_n[k*nq + r];
            accum[p*nq*nq + q*nq + r] += tmp;
        }


        // ==========================================
        // PHASE 2: Apply Scalar Metric
        // ==========================================
        // Single flattened loop for speed: (1 / |det J|) * quad_weight
        int e_offset = e * nq * nq * nq;
        for(unsigned int q_idx = 0; q_idx < nq * nq * nq; ++q_idx){
            accum[q_idx] *= G_scalar[e_offset + q_idx]; 
        }

        // ==========================================
        // PHASE 3: Project back to Nodes (Transpose)
        // ==========================================
        
        // --- Component 0 (Test with x-derivative) ---
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += accum[p*nq*nq + q*nq + r] * basis_t[k*nq + r];
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
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * dbasis_n[i*nq + p];
            out_0[i*nm_t*nm_t + j*nm_t + k] = tmp;
        }

        // --- Component 1 (Test with y-derivative) ---
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += accum[p*nq*nq + q*nq + r] * basis_t[k*nq + r];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * dbasis_n[j*nq + q];
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

        // --- Component 2 (Test with z-derivative) ---
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += accum[p*nq*nq + q*nq + r] * dbasis_n[k*nq + r];
            wsp1[p*nq*nm_n + q*nm_n + k]  = tmp;
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

    delete[] wsp0; delete[] wsp1; delete[] accum; 
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
#endif   //SERIAL_DIVDIV_OPERATOR_HPP