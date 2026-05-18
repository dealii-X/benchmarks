#ifndef SERIAL_MIXEDGRAD_OPERATOR_HPP
#define SERIAL_MIXEDGRAD_OPERATOR_HPP

#include <numeric>
#include <algorithm>

namespace Serial {

template<typename T>
T MixedGrad(const unsigned int nq, 
    const unsigned int nm_t, const unsigned int nm_n, const unsigned int nm_p,
    const unsigned int nelmt,
    const T *__restrict__ dbasis_n, const T *__restrict__ basis_t,
    const T *__restrict__ basis_p,
    const T *__restrict__ G_scalar, T *__restrict__ in_p, T * __restrict__ out_u)
{
    // DoFs per element for Raviart-Thomas velocity (output)
    const unsigned int ndof_u_1D = nm_n * nm_t * nm_t;
    const unsigned int ndof_u_total = ndof_u_1D + ndof_u_1D + ndof_u_1D;

    // DoFs per element for scalar pressure (input)
    const unsigned int ndof_p_total = nm_p * nm_p * nm_p;
  
    // Workspace arrays
    T *wsp0  = new T[nq * nq * nq];
    T *wsp1  = new T[nq * nq * nq];
    T *accum = new T[nq * nq * nq]; // Holds the scalar pressure at quad points

    for(unsigned int e = 0; e < nelmt; ++e){
        
        T* in_e = in_p + e * ndof_p_total;
        T* out_e = out_u + e * ndof_u_total;
        
        // Output velocity components
        T* out_0 = out_e;
        T* out_1 = out_e + ndof_u_1D;
        T* out_2 = out_e + ndof_u_1D + ndof_u_1D;

        std::fill(accum, accum + nq * nq * nq, (T)0);

        // ==========================================
        // PHASE 1: Interpolate Scalar Pressure to Quadrature Points
        // (Uses basis_p for all 3 directions)
        // ==========================================
        
        // --- Interpolate x-direction ---
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_p; ++j)
        for(unsigned int k=0; k<nm_p; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_p; ++i)
                tmp += in_e[i*nm_p*nm_p + j*nm_p + k] * basis_p[i*nq + p];
            wsp0[p*nm_p*nm_p + j*nm_p + k] = tmp;
        }

        // --- Interpolate y-direction ---
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_p; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm_p; ++j)
                tmp += wsp0[p*nm_p*nm_p + j*nm_p + k] * basis_p[j*nq + q];
            wsp1[p*nq*nm_p + q*nm_p + k] = tmp;
        }

        // --- Interpolate z-direction ---
        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_p; ++k)
                tmp += wsp1[p*nq*nm_p + q*nm_p + k] * basis_p[k*nq + r];
            accum[p*nq*nq + q*nq + r] = tmp;
        }

        // ==========================================
        // PHASE 2: Apply Scalar Metric
        // ==========================================
        // Weight the pressure by the quadrature weights (and Jacobian determinant 
        // cancellations due to Piola, exact same metric as HDivMixedDiv).
        int e_offset = e * nq * nq * nq;
        for(unsigned int q_idx = 0; q_idx < nq * nq * nq; ++q_idx){
            accum[q_idx] *= G_scalar[e_offset + q_idx]; 
        }

        // ==========================================
        // PHASE 3: Project out to H(div) Velocity Space
        // (Test against the divergence of the velocity basis functions)
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
    
    return std::transform_reduce(out_u, out_u + nelmt * ndof_u_total,
                          out_u, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}


}  //namespace Serial
#endif   //SERIAL_MIXEDGRAD_OPERATOR_HPP