#ifndef SERIAL_LINEAR_ELASTICITY_HPP
#define SERIAL_LINEAR_ELASTICITY_HPP

#include <numeric>
#include <algorithm>

namespace LinearElasticity{
namespace Serial {

template<typename T>
T SumFactorization(const unsigned int nq, 
    const unsigned int nm, 
    const unsigned int nelmt,
    const T *__restrict__ basis, const T *__restrict__ dbasis,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out,
    const T lambda = (T)1, const T mu = (T)1)
{
    const unsigned int ncomp = 3;
    const unsigned int ndof_1D = nm * nm * nm;
    const unsigned int ndof_total = ncomp * ndof_1D;

    T *wsp0 = new T[nq * nq * nq];
    T *wsp1 = new T[nq * nq * nq];
    T *uq   = new T[ncomp * nq * nq * nq];
    T *rqr  = new T[ncomp * nq * nq * nq];
    T *rqs  = new T[ncomp * nq * nq * nq];
    T *rqt  = new T[ncomp * nq * nq * nq];

    for(unsigned int e = 0; e < nelmt; ++e){
        
        const size_t element_offset = static_cast<size_t>(e) * ndof_1D;
        const size_t component_offset = static_cast<size_t>(nelmt) * ndof_1D;

        std::fill(uq,  uq  + ncomp * nq * nq * nq, (T)0);
        std::fill(rqr, rqr + ncomp * nq * nq * nq, (T)0);
        std::fill(rqs, rqs + ncomp * nq * nq * nq, (T)0);
        std::fill(rqt, rqt + ncomp * nq * nq * nq, (T)0);

        /*
        Interpolate to GL nodes (all 3 components required before quadrature coupling)
        */
        for(unsigned int c = 0; c < ncomp; ++c){

            const T* in_c = in + c * component_offset + element_offset;
            T* uq_c = uq + c * nq * nq * nq;

            std::fill(wsp0, wsp0 + nq * nq * nq, (T)0);
            std::fill(wsp1, wsp1 + nq * nq * nq, (T)0);

            //step-1 : Copy from in to the wsp0
            for(unsigned int i = 0; i < nm; i++){
                for(unsigned int j = 0; j < nm; j++){
                    for(unsigned int k = 0; k < nm; k++){
                        wsp0[i * nm * nm + j * nm + k] = in_c[i * nm * nm + j * nm + k];
                    }
                }
            }

            //step-2 : direction 0
            for(unsigned int p = 0; p < nq; p++){
                for(unsigned int k = 0; k < nm; k++){
                    for(unsigned int j = 0; j < nm; j++){
                        for(unsigned int i = 0; i < nm; i++){
                            wsp1[p * nm * nm + j * nm + k] += wsp0[i * nm * nm + j * nm + k] * basis[i * nq + p];
                        }
                    }
                }
            }
            std::fill(wsp0, wsp0 + nq * nq * nq, (T)0);

            //step-3 : direction 1
            for(unsigned int q = 0; q < nq; q++){
                for(unsigned int p = 0; p < nq; p++){
                    for(unsigned int k = 0; k < nm; k++){
                        for(unsigned int j = 0; j < nm; j++){
                            wsp0[q * nq * nm + p * nm + k] += wsp1[p * nm * nm + j * nm + k] * basis[j * nq + q];
                        }
                    }
                }
            }

            //step-4 : direction 2
            for(unsigned int r = 0; r < nq; r++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int k = 0; k < nm; k++){
                            uq_c[p * nq * nq + q * nq + r] += wsp0[q * nq * nm + p * nm + k] * basis[k * nq + r];
                        }
                    }
                }
            }
        }
        
        // Geometric vals (wdetJ and 3x3 inverse Jacobian J^{-1})
        T wdetJ, drdx, drdy, drdz, dsdx, dsdy, dsdz, dtdx, dtdy, dtdz;
        const size_t g_offset = static_cast<size_t>(e) * nq * nq * nq * 10;

        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){              
                for(unsigned int r = 0; r < nq; ++r){

                    const size_t q_idx = p * nq * nq + q * nq + r;

                    //step-5 : Load Geometric Factors
                    wdetJ = G[g_offset + 0 * nq * nq * nq + q_idx];
                    drdx  = G[g_offset + 1 * nq * nq * nq + q_idx];
                    drdy  = G[g_offset + 2 * nq * nq * nq + q_idx];
                    drdz  = G[g_offset + 3 * nq * nq * nq + q_idx];
                    dsdx  = G[g_offset + 4 * nq * nq * nq + q_idx];
                    dsdy  = G[g_offset + 5 * nq * nq * nq + q_idx];
                    dsdz  = G[g_offset + 6 * nq * nq * nq + q_idx];
                    dtdx  = G[g_offset + 7 * nq * nq * nq + q_idx];
                    dtdy  = G[g_offset + 8 * nq * nq * nq + q_idx];
                    dtdz  = G[g_offset + 9 * nq * nq * nq + q_idx];
                    
                    //step-6 : Multiply by quadrature-space D
                    T ur[3] = {(T)0, (T)0, (T)0};
                    T us[3] = {(T)0, (T)0, (T)0};
                    T ut[3] = {(T)0, (T)0, (T)0};

                    for(unsigned int c = 0; c < ncomp; ++c){
                        const T* uq_c = uq + c * nq * nq * nq;

                        for(unsigned int n = 0; n < nq; ++n){
                            ur[c] += uq_c[n * nq * nq + q * nq + r] * dbasis[n * nq + p];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            us[c] += uq_c[p * nq * nq + n * nq + r] * dbasis[n * nq + q];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            ut[c] += uq_c[p * nq * nq + q * nq + n] * dbasis[n * nq + r];
                        }
                    }

                    // step-7 : Apply chain rule and constitutive law (eps : C : eps)
                    T dudx[3][3];
                    for(unsigned int c = 0; c < ncomp; ++c){
                        dudx[c][0] = ur[c] * drdx + us[c] * dsdx + ut[c] * dtdx;
                        dudx[c][1] = ur[c] * drdy + us[c] * dsdy + ut[c] * dtdy;
                        dudx[c][2] = ur[c] * drdz + us[c] * dsdz + ut[c] * dtdz;
                    }

                    const T div_u = dudx[0][0] + dudx[1][1] + dudx[2][2];

                    const T sigma_xx = lambda * div_u + (T)2 * mu * dudx[0][0];
                    const T sigma_yy = lambda * div_u + (T)2 * mu * dudx[1][1];
                    const T sigma_zz = lambda * div_u + (T)2 * mu * dudx[2][2];
                    const T sigma_xy = mu * (dudx[0][1] + dudx[1][0]);
                    const T sigma_xz = mu * (dudx[0][2] + dudx[2][0]);
                    const T sigma_yz = mu * (dudx[1][2] + dudx[2][1]);

                    const T sigma[3][3] = {
                        {sigma_xx, sigma_xy, sigma_xz},
                        {sigma_xy, sigma_yy, sigma_yz},
                        {sigma_xz, sigma_yz, sigma_zz}
                    };

                    for(unsigned int c = 0; c < ncomp; ++c){
                        T* rqr_c = rqr + c * nq * nq * nq;
                        T* rqs_c = rqs + c * nq * nq * nq;
                        T* rqt_c = rqt + c * nq * nq * nq;

                        rqr_c[q_idx] = wdetJ * (sigma[c][0] * drdx + sigma[c][1] * drdy + sigma[c][2] * drdz);
                        rqs_c[q_idx] = wdetJ * (sigma[c][0] * dsdx + sigma[c][1] * dsdy + sigma[c][2] * dsdz);
                        rqt_c[q_idx] = wdetJ * (sigma[c][0] * dtdx + sigma[c][1] * dtdy + sigma[c][2] * dtdz);
                    }
                }
            }
        }

        for(unsigned int c = 0; c < ncomp; ++c){

            T* out_c = out + c * component_offset + element_offset;
            const T* rqr_c = rqr + c * nq * nq * nq;
            const T* rqs_c = rqs + c * nq * nq * nq;
            const T* rqt_c = rqt + c * nq * nq * nq;

            std::fill(wsp1, wsp1 + nq * nq * nq, (T)0);

            // step-8 : Compute out vector in GL nodes
            for(unsigned int p = 0; p < nq; ++p){                      
                for(unsigned int q = 0; q < nq; ++q){ 
                    for(unsigned int r = 0; r < nq; ++r){ 

                        T tmp0 = (T)0;
                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += rqr_c[n * nq * nq + q * nq + r] * dbasis[p * nq + n];

                        for(unsigned int n = 0; n < nq; ++n)                
                            tmp0 += rqs_c[p * nq * nq + n * nq + r] * dbasis[q * nq + n];

                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += rqt_c[p * nq * nq + q * nq + n] * dbasis[r * nq + n];

                        wsp1[p * nq * nq + q * nq + r] = tmp0;
                    }
                }
            }


            /*
            Interpolate to GLL nodes
            */

            //step-9 : direction 2
            std::fill(wsp0, wsp0 + nq * nq * nq, (T)0);

            for(unsigned int k = 0; k < nm; k++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){ 
                        for(unsigned int r = 0; r < nq; r++){
                            wsp0[q * nq * nm + p * nm + k] += wsp1[p * nq * nq + q * nq + r] * basis[k * nq + r];
                        }
                    }
                }
            }
            std::fill(wsp1, wsp1 + nq * nq * nq, (T)0);

            //step-10 : direction 1
            for(unsigned int j = 0; j < nm; j++){
                for(unsigned int k = 0; k < nm; k++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int q = 0; q < nq; q++){
                            wsp1[p * nm * nm + j * nm + k] += wsp0[q * nq * nm + p * nm + k] * basis[j * nq + q];
                        }
                    }
                }
            }
            std::fill(wsp0, wsp0 + nq * nq * nq, (T)0);

            //step-11 : direction 0
            for(unsigned int i = 0; i < nm; i++){
                for(unsigned int j = 0; j < nm; j++){
                    for(unsigned int k = 0; k < nm; k++){
                        for(unsigned int p = 0; p < nq; p++){
                            wsp0[i * nm * nm + j * nm + k] += wsp1[p * nm * nm + j * nm + k] * basis[i * nq + p];
                        }
                    }
                }
            }

            //step-12 : Copy from wsp0 to out
            for(unsigned int i = 0; i < nm; i++){
                for(unsigned int j = 0; j < nm; j++){
                    for(unsigned int k = 0; k < nm; k++){
                        out_c[i * nm * nm + j * nm + k] = wsp0[i * nm * nm + j * nm + k];
                    }
                }
            }
        }
    }

    delete[] wsp0; delete[] wsp1; delete[] uq; delete[] rqr; delete[] rqs; delete[] rqt;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
}

#endif   //SERIAL_LINEAR_ELASTICITY_HPP