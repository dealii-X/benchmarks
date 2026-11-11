#ifndef SERIAL_STIFFNESS_OPERATOR_HPP
#define SERIAL_STIFFNESS_OPERATOR_HPP

#include <numeric>
#include <algorithm>

namespace Serial {

template<typename T>
T Stiffness(const unsigned int nq, 
    const unsigned int nm_t, const unsigned int nm_n, 
    const unsigned int nelmt,
    const T *__restrict__ basis_n, const T *__restrict__ basis_t,
    const T *__restrict__ dbasis_n, const T *__restrict__ dbasis_t,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out)
{
    // Total DoFs per element for a Raviart-Thomas vector
    const unsigned int ndof_1D = nm_n * nm_t * nm_t;
    const unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

    // Intermediate vals
    T *wsp0 = new T[nq * nq * nq];
    T *wsp1 = new T[nq * nq * nq];
    
    // Arrays to hold the 9 reference gradient components at quadrature points
    T *uq_0_x = new T[nq * nq * nq];
    T *uq_0_y = new T[nq * nq * nq];
    T *uq_0_z = new T[nq * nq * nq];

    T *uq_1_x = new T[nq * nq * nq];
    T *uq_1_y = new T[nq * nq * nq];
    T *uq_1_z = new T[nq * nq * nq];

    T *uq_2_x = new T[nq * nq * nq];
    T *uq_2_y = new T[nq * nq * nq];
    T *uq_2_z = new T[nq * nq * nq];


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
        

        // ============================================================
        // PHASE 1: Interpolate & Differentiate to Quadrature Nodes
        // ============================================================
        
        // --- Component 0 (x-direction) ---
        // x is normal (basis_n, dbasis_n), y and z are tangent (basis_t, dbasis_t)
        
        // 1a. Interpolate x (basis_n) -> shared for d/dy and d/dz
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_n; ++i)
                tmp += in_0[i*nm_t*nm_t + j*nm_t + k] * basis_n[i*nq + p];
            wsp0[p*nm_t*nm_t + j*nm_t + k] = tmp;
        }

        // Interpolate y (basis_t), then differentiate z (dbasis_t) -> uq_0_z
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
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * dbasis_t[k*nq + r];
            uq_0_z[r*nq*nq + q*nq + p] = tmp;
        }

        // Differentiate y (dbasis_t), then interpolate z (basis_t) -> uq_0_y
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = (T)0;
            for(unsigned int j=0; j<nm_t; ++j)
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * dbasis_t[j*nq + q];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_t; ++k)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * basis_t[k*nq + r];
            uq_0_y[r*nq*nq + q*nq + p] = tmp;
        }

        // 1b. Differentiate x (dbasis_n), interpolate y (basis_t), interpolate z (basis_t) -> uq_0_x
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
            uq_0_x[r*nq*nq + q*nq + p] = tmp;
        }


        // --- Component 1 (y-direction) ---
        // y is normal (basis_n, dbasis_n), x and z are tangent (basis_t, dbasis_t)

        // 1a. Interpolate x (basis_t) -> shared for d/dy and d/dz
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_1[i*nm_n*nm_t + j*nm_t + k] * basis_t[i*nq + p];
            wsp0[p*nm_n*nm_t + j*nm_t + k] = tmp;
        }

        // Interpolate y (basis_n), then differentiate z (dbasis_t) -> uq_1_z
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
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * dbasis_t[k*nq + r];
            uq_1_z[r*nq*nq + q*nq + p] = tmp;
        }

        // Differentiate y (dbasis_n), then interpolate z (basis_t) -> uq_1_y
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
            uq_1_y[r*nq*nq + q*nq + p] = tmp;
        }

        // 1b. Differentiate x (dbasis_t), interpolate y (basis_n), interpolate z (basis_t) -> uq_1_x
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_1[i*nm_n*nm_t + j*nm_t + k] * dbasis_t[i*nq + p];
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
            uq_1_x[r*nq*nq + q*nq + p] = tmp;
        }


        // --- Component 2 (z-direction) ---
        // z is normal (basis_n, dbasis_n), x and y are tangent (basis_t, dbasis_t)

        // 1a. Interpolate x (basis_t) -> shared for d/dy and d/dz
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_2[i*nm_t*nm_n + j*nm_n + k] * basis_t[i*nq + p];
            wsp0[p*nm_t*nm_n + j*nm_n + k] = tmp;
        }

        // Interpolate y (basis_t), then differentiate z (dbasis_n) -> uq_2_z
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
            uq_2_z[r*nq*nq + q*nq + p] = tmp;
        }

        // Differentiate y (dbasis_t), then interpolate z (basis_n) -> uq_2_y
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0;
            for(unsigned int j=0; j<nm_t; ++j)
               tmp += wsp0[p*nm_t*nm_n + j*nm_n + k] * dbasis_t[j*nq + q];
            wsp1[p*nq*nm_n + q*nm_n + k] = tmp;
        }

        for(unsigned int r=0; r<nq; ++r)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int k=0; k<nm_n; ++k)
               tmp += wsp1[p*nq*nm_n + q*nm_n + k] * basis_n[k*nq + r];
            uq_2_y[r*nq*nq + q*nq + p] = tmp;
        }

        // 1b. Differentiate x (dbasis_t), interpolate y (basis_t), interpolate z (basis_n) -> uq_2_x
        for(unsigned int p=0; p<nq; ++p)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0;
            for(unsigned int i=0; i<nm_t; ++i)
                tmp += in_2[i*nm_t*nm_n + j*nm_n + k] * dbasis_t[i*nq + p];
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
            uq_2_x[r*nq*nq + q*nq + p] = tmp;
        }


        // ============================================================
        // PHASE 2: Apply Geometry Metric at Quadrature Points
        // ============================================================
        for(unsigned int r = 0; r < nq; ++r){
            for(unsigned int q = 0; q < nq; ++q){              
                for(unsigned int p = 0; p < nq; ++p){

                    int q_idx = r * nq * nq + q * nq + p;
                    int e_offset = e * 6 * nq * nq * nq;

                    // Load Symmetric 3x3 Metric Factors (w_q * nu * |J| * J^{-1} * J^{-T})
                    T G00 = G[e_offset + 0 * nq*nq*nq + q_idx];
                    T G01 = G[e_offset + 1 * nq*nq*nq + q_idx];
                    T G02 = G[e_offset + 2 * nq*nq*nq + q_idx];
                    T G11 = G[e_offset + 3 * nq*nq*nq + q_idx];
                    T G12 = G[e_offset + 4 * nq*nq*nq + q_idx];
                    T G22 = G[e_offset + 5 * nq*nq*nq + q_idx];

                    // Load the 9 reference velocity gradient components
                    T u0x = uq_0_x[q_idx], u0y = uq_0_y[q_idx], u0z = uq_0_z[q_idx];
                    T u1x = uq_1_x[q_idx], u1y = uq_1_y[q_idx], u1z = uq_1_z[q_idx];
                    T u2x = uq_2_x[q_idx], u2y = uq_2_y[q_idx], u2z = uq_2_z[q_idx];

                    // Linear metric contraction
                    uq_0_x[q_idx] = G00 * u0x + G01 * u0y + G02 * u0z;
                    uq_0_y[q_idx] = G01 * u0x + G11 * u0y + G12 * u0z;
                    uq_0_z[q_idx] = G02 * u0x + G12 * u0y + G22 * u0z;

                    uq_1_x[q_idx] = G00 * u1x + G01 * u1y + G02 * u1z;
                    uq_1_y[q_idx] = G01 * u1x + G11 * u1y + G12 * u1z;
                    uq_1_z[q_idx] = G02 * u1x + G12 * u1y + G22 * u1z;

                    uq_2_x[q_idx] = G00 * u2x + G01 * u2y + G02 * u2z;
                    uq_2_y[q_idx] = G01 * u2x + G11 * u2y + G12 * u2z;
                    uq_2_z[q_idx] = G02 * u2x + G12 * u2y + G22 * u2z;

                }
            }
        }


        // ============================================================
        // PHASE 3: Project back to Nodes (Transpose)
        // ============================================================
        
        // --- Component 0 (x-direction) ---
        // 3a. Terms sharing basis_n^T in x (uq_0_z and uq_0_y)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_0_z[r*nq*nq + q*nq + p] * dbasis_t[k*nq + r];
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

        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_0_y[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * dbasis_t[j*nq + q];
            wsp0[p*nm_t*nm_t + j*nm_t + k] += tmp;
        }

        for(unsigned int i=0; i<nm_n; ++i)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_t*nm_t + j*nm_t + k] * basis_n[i*nq + p];
            out_0[i*nm_t*nm_t + j*nm_t + k] = tmp;
        }

        // 3b. Term using dbasis_n^T in x (uq_0_x)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_0_x[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
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
            out_0[i*nm_t*nm_t + j*nm_t + k] += tmp;
        }


        // --- Component 1 (y-direction) ---
        // Normal in y (basis_n, dbasis_n), tangent in x and z (basis_t, dbasis_t)
        // 3a. Terms sharing basis_t^T in x (uq_1_z and uq_1_y)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_1_z[r*nq*nq + q*nq + p] * dbasis_t[k*nq + r];
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

        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_1_y[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
            wsp1[p*nq*nm_t + q*nm_t + k] = tmp;
        }

        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_t + q*nm_t + k] * dbasis_n[j*nq + q];
            wsp0[p*nm_n*nm_t + j*nm_t + k] += tmp;
        }

        for(unsigned int i=0; i<nm_t; ++i)
        for(unsigned int j=0; j<nm_n; ++j)
        for(unsigned int k=0; k<nm_t; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_n*nm_t + j*nm_t + k] * basis_t[i*nq + p];
            out_1[i*nm_n*nm_t + j*nm_t + k] = tmp;
        }

        // 3b. Term using dbasis_t^T in x (uq_1_x)
        for(unsigned int k=0; k<nm_t; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_1_x[r*nq*nq + q*nq + p] * basis_t[k*nq + r];
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
                tmp += wsp0[p*nm_n*nm_t + j*nm_t + k] * dbasis_t[i*nq + p];
            out_1[i*nm_n*nm_t + j*nm_t + k] += tmp;
        }


        // --- Component 2 (z-direction) ---
        // Normal in z (basis_n, dbasis_n), tangent in x and y (basis_t, dbasis_t)
        // 3a. Terms sharing basis_t^T in x (uq_2_z and uq_2_y)
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_2_z[r*nq*nq + q*nq + p] * dbasis_n[k*nq + r];
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

        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_2_y[r*nq*nq + q*nq + p] * basis_n[k*nq + r];
            wsp1[p*nq*nm_n + q*nm_n + k] = tmp;
        }

        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int q=0; q<nq; ++q)
                tmp += wsp1[p*nq*nm_n + q*nm_n + k] * dbasis_t[j*nq + q];
            wsp0[p*nm_t*nm_n + j*nm_n + k] += tmp;
        }

        for(unsigned int i=0; i<nm_t; ++i)
        for(unsigned int j=0; j<nm_t; ++j)
        for(unsigned int k=0; k<nm_n; ++k){
            T tmp = 0.0;
            for(unsigned int p=0; p<nq; ++p)
                tmp += wsp0[p*nm_t*nm_n + j*nm_n + k] * basis_t[i*nq + p];
            out_2[i*nm_t*nm_n + j*nm_n + k] = tmp;
        }

        // 3b. Term using dbasis_t^T in x (uq_2_x)
        for(unsigned int k=0; k<nm_n; ++k)
        for(unsigned int q=0; q<nq; ++q)
        for(unsigned int p=0; p<nq; ++p){
            T tmp = 0;
            for(unsigned int r=0; r<nq; ++r)
                tmp += uq_2_x[r*nq*nq + q*nq + p] * basis_n[k*nq + r];
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
                tmp += wsp0[p*nm_t*nm_n + j*nm_n + k] * dbasis_t[i*nq + p];
            out_2[i*nm_t*nm_n + j*nm_n + k] += tmp;
        }

    }

    delete[] wsp0; delete[] wsp1;
    delete[] uq_0_x; delete[] uq_0_y; delete[] uq_0_z;
    delete[] uq_1_x; delete[] uq_1_y; delete[] uq_1_z;
    delete[] uq_2_x; delete[] uq_2_y; delete[] uq_2_z;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
#endif   //SERIAL_STIFFNESS_OPERATOR_HPP