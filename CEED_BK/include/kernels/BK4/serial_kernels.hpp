#ifndef BK4_SERIALKERNELS_HPP
#define BK4_SERIALKERNELS_HPP

#include <numeric>
#include <algorithm>

namespace BK4{
namespace Serial {

template<typename T>
T SumFactorization(const unsigned int nq, 
    const unsigned int nm, 
    const unsigned int nelmt,
    const T *__restrict__ basis, const T *__restrict__ dbasis,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out)
{
    // Total DoFs per element for a 3D vector field (3 components)
    const unsigned int ndof_1D = nm * nm * nm;
    const unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

    // Intermediate vals
    T *wsp0 = new T[nq * nq * nq];
    T *wsp1 = new T[nq * nq * nq];
    
    // Arrays to hold the 3 vector component derivatives at quadrature points
    T *g0_0 = new T[nq * nq * nq];
    T *g0_1 = new T[nq * nq * nq];
    T *g0_2 = new T[nq * nq * nq];

    T *g1_0 = new T[nq * nq * nq];
    T *g1_1 = new T[nq * nq * nq];
    T *g1_2 = new T[nq * nq * nq];

    T *g2_0 = new T[nq * nq * nq];
    T *g2_1 = new T[nq * nq * nq];
    T *g2_2 = new T[nq * nq * nq];


    for(unsigned int e = 0; e < nelmt; ++e){
        
        const size_t element_offset = static_cast<size_t>(e) * ndof_1D;

        const size_t component_offset = static_cast<size_t>(nelmt) * ndof_1D;

        const T* in_0 = in + element_offset;

        const T* in_1 = in + component_offset + element_offset;

        const T* in_2 = in + 2 * component_offset + element_offset;

        T* out_0 = out + element_offset;

        T* out_1 = out + component_offset + element_offset;

        T* out_2 = out + 2 * component_offset + element_offset;
        

        // ==========================================
        // PHASE 1: Differentiate to Quadrature Nodes
        // ==========================================
        
        // --- Component 0 (x-direction) ---
        // derivative in x (direction 0)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_0[i*nm*nm + j*nm + k] * dbasis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = (T)0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g0_0[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in y (direction 1)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_0[i*nm*nm + j*nm + k] * basis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g0_1[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in z (direction 2)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[k*nq + r];
            g0_2[r*nq*nq + q*nq + p] = tmp;
        }

        // --- Component 1 (y-direction) ---
        // derivative in x (direction 0)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_1[i*nm*nm + j*nm + k] * dbasis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g1_0[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in y (direction 1)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_1[i*nm*nm + j*nm + k] * basis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g1_1[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in z (direction 2)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[k*nq + r];
            g1_2[r*nq*nq + q*nq + p] = tmp;
        }

        // --- Component 2 (z-direction) ---
        // derivative in x (direction 0)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_2[i*nm*nm + j*nm + k] * dbasis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g2_0[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in y (direction 1)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm; ++i)
                tmp += in_2[i*nm*nm + j*nm + k] * basis[i*nq + p];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[k*nq + r];
            g2_1[r*nq*nq + q*nq + p] = tmp;
        }

        // derivative in z (direction 2)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm; ++j)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[j*nq + q];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm; ++k)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[k*nq + r];
            g2_2[r*nq*nq + q*nq + p] = tmp;
        }


        // ==========================================
        // PHASE 2: Apply Geometry Metric Tensor
        // ==========================================
        for(unsigned int r = 0; r < nq; ++r){
            for(unsigned int q = 0; q < nq; ++q){              
                for(unsigned int p = 0; p < nq; ++p){

                    const unsigned int q_idx = r * nq * nq + q * nq + p;

                    // G is stored as [p][q][r]
                    const unsigned int G_idx = p * nq * nq + q * nq + r;

                    const unsigned int e_offset = e * 6 * nq * nq * nq;

                    T G00 = G[e_offset + 0 * nq*nq*nq + G_idx];
                    T G01 = G[e_offset + 1 * nq*nq*nq + G_idx];
                    T G02 = G[e_offset + 2 * nq*nq*nq + G_idx];
                    T G11 = G[e_offset + 3 * nq*nq*nq + G_idx];
                    T G12 = G[e_offset + 4 * nq*nq*nq + G_idx];
                    T G22 = G[e_offset + 5 * nq*nq*nq + G_idx];

                    // Component 0
                    T g0_0_val = g0_0[q_idx];
                    T g0_1_val = g0_1[q_idx];
                    T g0_2_val = g0_2[q_idx];

                    g0_0[q_idx] = G00 * g0_0_val + G01 * g0_1_val + G02 * g0_2_val;
                    g0_1[q_idx] = G01 * g0_0_val + G11 * g0_1_val + G12 * g0_2_val;
                    g0_2[q_idx] = G02 * g0_0_val + G12 * g0_1_val + G22 * g0_2_val;

                    // Component 1
                    T g1_0_val = g1_0[q_idx];
                    T g1_1_val = g1_1[q_idx];
                    T g1_2_val = g1_2[q_idx];

                    g1_0[q_idx] = G00 * g1_0_val + G01 * g1_1_val + G02 * g1_2_val;
                    g1_1[q_idx] = G01 * g1_0_val + G11 * g1_1_val + G12 * g1_2_val;
                    g1_2[q_idx] = G02 * g1_0_val + G12 * g1_1_val + G22 * g1_2_val;

                    // Component 2
                    T g2_0_val = g2_0[q_idx];
                    T g2_1_val = g2_1[q_idx];
                    T g2_2_val = g2_2[q_idx];

                    g2_0[q_idx] = G00 * g2_0_val + G01 * g2_1_val + G02 * g2_2_val;
                    g2_1[q_idx] = G01 * g2_0_val + G11 * g2_1_val + G12 * g2_2_val;
                    g2_2[q_idx] = G02 * g2_0_val + G12 * g2_1_val + G22 * g2_2_val;
                }
            }
        }


        // ==========================================
        // PHASE 3: Project back to Nodes (Transpose)
        // ==========================================
        
        // --- Component 0 (x-direction) ---
        // Term 0: Transpose of (D, B, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g0_0[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[i*nq + p];
            out_0[i*nm*nm + j*nm + k] = tmp;
        }

        // Term 1: Transpose of (B, D, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g0_1[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_0[i*nm*nm + j*nm + k] += tmp;
        }

        // Term 2: Transpose of (B, B, D)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g0_2[r*nq*nq + q*nq + p] * dbasis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_0[i*nm*nm + j*nm + k] += tmp;
        }

        // --- Component 1 (y-direction) ---
        // Term 0: Transpose of (D, B, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g1_0[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[i*nq + p];
            out_1[i*nm*nm + j*nm + k] = tmp;
        }

        // Term 1: Transpose of (B, D, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g1_1[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_1[i*nm*nm + j*nm + k] += tmp;
        }

        // Term 2: Transpose of (B, B, D)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g1_2[r*nq*nq + q*nq + p] * dbasis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_1[i*nm*nm + j*nm + k] += tmp;
        }

        // --- Component 2 (z-direction) ---
        // Term 0: Transpose of (D, B, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g2_0[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * dbasis[i*nq + p];
            out_2[i*nm*nm + j*nm + k] = tmp;
        }

        // Term 1: Transpose of (B, D, B)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g2_1[r*nq*nq + q*nq + p] * basis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * dbasis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_2[i*nm*nm + j*nm + k] += tmp;
        }

        // Term 2: Transpose of (B, B, D)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += g2_2[r*nq*nq + q*nq + p] * dbasis[k*nq + r];
            wsp1[p*nq*nm + q*nm + k] = tmp;
        }

        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm + q*nm + k] * basis[j*nq + q];
            wsp0[p*nm*nm + j*nm + k] = tmp;
        }

        for(unsigned int i=0; i<nm; ++i)
        for(unsigned int j=0; j<nm; ++j)
        for(unsigned int k=0; k<nm; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm*nm + j*nm + k] * basis[i*nq + p];
            out_2[i*nm*nm + j*nm + k] += tmp;
        }

    }

    delete[] wsp0; delete[] wsp1; 
    delete[] g0_0; delete[] g0_1; delete[] g0_2;
    delete[] g1_0; delete[] g1_1; delete[] g1_2;
    delete[] g2_0; delete[] g2_1; delete[] g2_2;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
}

#endif   //BK3_SERIALKERNELS_HPP