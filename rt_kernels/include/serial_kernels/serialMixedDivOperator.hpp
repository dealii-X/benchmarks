#ifndef SERIAL_MIXEDDIV_OPERATOR_HPP
#define SERIAL_MIXEDDIV_OPERATOR_HPP

#include <numeric>
#include <algorithm>

namespace Serial {

template<typename T>
T MixedDiv(const unsigned int nq, 
    const unsigned int nm_t, const unsigned int nm_n, const unsigned int nm_p,
    const unsigned int nelmt,
    const T *__restrict__ dbasis_n, const T *__restrict__ basis_t,
    const T *__restrict__ basis_p,
    const T *__restrict__ G_scalar, T *__restrict__ in_u, T * __restrict__ out_p)

{
    // DoFs per element for Raviart-Thomas velocity
    const unsigned int ndof_u_1D = nm_n * nm_t * nm_t;
    const unsigned int ndof_u_total = ndof_u_1D + ndof_u_1D + ndof_u_1D;

    // DoFs per element for scalar pressure (e.g., DG polynomial space)

    const unsigned int ndof_p_total = nm_p * nm_p * nm_p;

    T *wsp0  = new T[nq * nq * nq];
    T *wsp1  = new T[nq * nq * nq];
    T *accum = new T[nq * nq * nq]; // Holds the total scalar divergence at quad points


    for(unsigned int e = 0; e < nelmt; ++e){

        T* in_e = in_u + e * ndof_u_total;
        T* out_e = out_p + e * ndof_p_total;

        T* in_0 = in_e;
        T* in_1 = in_e + ndof_u_1D;
        T* in_2 = in_e + ndof_u_1D + ndof_u_1D;

        std::fill(accum, accum + nq * nq * nq, (T)0);

        // ==========================================
        // PHASE 1: Compute Reference Divergence
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

        int e_offset = e * nq * nq * nq;
        for(unsigned int q_idx = 0; q_idx < nq * nq * nq; ++q_idx){
            accum[q_idx] *= G_scalar[e_offset + q_idx]; 
        }

        // ==========================================
        // PHASE 3: Project to Scalar Pressure Space
        // (Test against pressure basis_p)
        // ==========================================

        for(unsigned int k=0; k<nm_p; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += accum[p*nq*nq + q*nq + r] * basis_p[k*nq + r];
            wsp1[p*nq*nm_p + q*nm_p + k] = tmp;
        }


        for(unsigned int j=0; j<nm_p; ++j)
        for(unsigned int k=0; k<nm_p; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_p + q*nm_p + k] * basis_p[j*nq + q];
            wsp0[p*nm_p*nm_p + j*nm_p + k] = tmp;
        }


        for(unsigned int i=0; i<nm_p; ++i)
        for(unsigned int j=0; j<nm_p; ++j)
        for(unsigned int k=0; k<nm_p; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_p*nm_p + j*nm_p + k] * basis_p[i*nq + p];
            out_e[i*nm_p*nm_p + j*nm_p + k] = tmp;
        }
    }

    delete[] wsp0; delete[] wsp1; delete[] accum; 

    
    // Return a dummy L2 norm style checksum

    return std::transform_reduce(out_p, out_p + nelmt * ndof_p_total,
                          out_p, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}


}  //namespace Serial
#endif   //SERIAL_MIXEDDIV_OPERATOR_HPP
