#ifndef BK3_SERIALKERNELS_OTF_HPP
#define BK3_SERIALKERNELS_OTF_HPP

#include <numeric>
#include <algorithm>
#include <vector>

namespace BK3{
namespace Serial{


template<typename T>
T SumFactorization_OTF(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1, const T *__restrict__ basis2,
    const T *__restrict__ dbasis0, const T *__restrict__ dbasis1, const T *__restrict__ dbasis2,
    const T *__restrict__ weights,
    const T *__restrict__ coord_q,
    T *__restrict__ in, T *__restrict__ out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;
    const unsigned int nquad = nq0 * nq1 * nq2;

    std::vector<T> wsp0_vec(std::max(nm0 * nm1 * nm2, nquad));
    std::vector<T> wsp1_vec(nquad);
    std::vector<T> rqr_vec(nquad);
    std::vector<T> rqs_vec(nquad);
    std::vector<T> rqt_vec(nquad);

    T* __restrict__ wsp0 = wsp0_vec.data();
    T* __restrict__ wsp1 = wsp1_vec.data();
    T* __restrict__ rqr  = rqr_vec.data();
    T* __restrict__ rqs  = rqs_vec.data();
    T* __restrict__ rqt  = rqt_vec.data();


    for(unsigned int e = 0; e < nelmt; ++e){

        // step-1 : Copy from in to wsp0
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    wsp0[i * nm1 * nm2 + j * nm2 + k] =
                        in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }

        // step-2 : direction 0
        for(unsigned int p = 0; p < nq0; p++){
            // i = 0 (Init)
            {
                const T b_val = basis0[0 * nq0 + p];
                for(unsigned int j = 0; j < nm1; j++){
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] = wsp0[0 * nm1 * nm2 + j * nm2 + k] * b_val;
                    }
                }
            }

            for(unsigned int i = 1; i < nm0; i++){
                const T b_val = basis0[i * nq0 + p];
                for(unsigned int j = 0; j < nm1; j++){
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[i * nm1 * nm2 + j * nm2 + k] * b_val;
                    }
                }
            }
        }

        // step-3 : direction 1
        for(unsigned int p = 0; p < nq0; p++){
            for(unsigned int q = 0; q < nq1; q++){
                {
                    const T b_val = basis1[0 * nq1 + q];
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp0[p * nq1 * nm2 + q * nm2 + k] = wsp1[p * nm1 * nm2 + 0 * nm2 + k] * b_val;
                    }
                }
                for(unsigned int j = 1; j < nm1; j++){
                    const T b_val = basis1[j * nq1 + q];
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp0[p * nq1 * nm2 + q * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * b_val;
                    }
                }
            }
        }

        // step-4 : direction 2
        for(unsigned int p = 0; p < nq0; p++){
            for(unsigned int q = 0; q < nq1; q++){
                // k = 0 (Init)
                for(unsigned int r = 0; r < nq2; r++){
                    wsp1[p * nq1 * nq2 + q * nq2 + r] = wsp0[p * nq1 * nm2 + q * nm2 + 0] * basis2[0 * nq2 + r];
                }
                // k > 0 (Accumulate)
                for(unsigned int k = 1; k < nm2; k++){
                    const T w0_val = wsp0[p * nq1 * nm2 + q * nm2 + k];
                    for(unsigned int r = 0; r < nq2; r++){
                        wsp1[p * nq1 * nq2 + q * nq2 + r] += w0_val * basis2[k * nq2 + r];
                    }
                }
            }
        }

        // ----------------------------------------------------------
        // Geometric action on the fly
        // ----------------------------------------------------------
        const unsigned int coord_base = e * 3 * nquad;
        const unsigned int xbase = coord_base;
        const unsigned int ybase = coord_base + nquad;
        const unsigned int zbase = coord_base + 2 * nquad;

        for(unsigned int p = 0; p < nq0; ++p){
            for(unsigned int q = 0; q < nq1; ++q){
                for(unsigned int r = 0; r < nq2; ++r){

                    T qr = 0.0, qs = 0.0, qt = 0.0;
                    for(unsigned int n = 0; n < nq0; ++n) qr += wsp1[n * nq1 * nq2 + q * nq2 + r] * dbasis0[n * nq0 + p];
                    for(unsigned int n = 0; n < nq1; ++n) qs += wsp1[p * nq1 * nq2 + n * nq2 + r] * dbasis1[n * nq1 + q];
                    for(unsigned int n = 0; n < nq2; ++n) qt += wsp1[p * nq1 * nq2 + q * nq2 + n] * dbasis2[n * nq2 + r];

                    T J00 = 0.0, J01 = 0.0, J02 = 0.0;
                    T J10 = 0.0, J11 = 0.0, J12 = 0.0;
                    T J20 = 0.0, J21 = 0.0, J22 = 0.0;

                    for(unsigned int n = 0; n < nq0; ++n){
                        const T x_r_n = coord_q[xbase + n * nq1 * nq2 + q * nq2 + r];
                        const T x_s_n = coord_q[xbase + p * nq1 * nq2 + n * nq2 + r];
                        const T x_t_n = coord_q[xbase + p * nq1 * nq2 + q * nq2 + n];

                        J00 += dbasis0[n * nq0 + p] * x_r_n;
                        J01 += dbasis1[n * nq1 + q] * x_s_n;
                        J02 += dbasis2[n * nq2 + r] * x_t_n;

                        const T y_r_n = coord_q[ybase + n * nq1 * nq2 + q * nq2 + r];
                        const T y_s_n = coord_q[ybase + p * nq1 * nq2 + n * nq2 + r];
                        const T y_t_n = coord_q[ybase + p * nq1 * nq2 + q * nq2 + n];

                        J10 += dbasis0[n * nq0 + p] * y_r_n;
                        J11 += dbasis1[n * nq1 + q] * y_s_n;
                        J12 += dbasis2[n * nq2 + r] * y_t_n;

                        const T z_r_n = coord_q[zbase + n * nq1 * nq2 + q * nq2 + r];
                        const T z_s_n = coord_q[zbase + p * nq1 * nq2 + n * nq2 + r];
                        const T z_t_n = coord_q[zbase + p * nq1 * nq2 + q * nq2 + n];

                        J20 += dbasis0[n * nq0 + p] * z_r_n;
                        J21 += dbasis1[n * nq1 + q] * z_s_n;
                        J22 += dbasis2[n * nq2 + r] * z_t_n;
                    }

                    // step-7 : Cofactor matrix C = det(J) J^{-T}
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

                    // step-8 : Direct factored metric action
                    const T t0 = C00 * qt + C10 * qs + C20 * qr;
                    const T t1 = C01 * qt + C11 * qs + C21 * qr;
                    const T t2 = C02 * qt + C12 * qs + C22 * qr;

                    const unsigned int pt = p * nq1 * nq2 + q * nq2 + r;
                    rqr[pt] = scale * (C00 * t0 + C01 * t1 + C02 * t2);
                    rqs[pt] = scale * (C10 * t0 + C11 * t1 + C12 * t2);
                    rqt[pt] = scale * (C20 * t0 + C21 * t1 + C22 * t2);
                }
            }
        }

        // ----------------------------------------------------------
        // step-9 : Compute out vector in GL nodes (Split & Coalesced)
        // ----------------------------------------------------------
        
        // 9a: D0^T rqr (Initializes wsp1)
        for(unsigned int n = 0; n < nq0; ++n){
            for(unsigned int p = 0; p < nq0; ++p){
                const T db = dbasis0[p * nq0 + n];
                for(unsigned int q = 0; q < nq1; ++q){
                    for(unsigned int r = 0; r < nq2; ++r){
                        if (n == 0) wsp1[p * nq1 * nq2 + q * nq2 + r]  = rqr[0 * nq1 * nq2 + q * nq2 + r] * db;
                        else        wsp1[p * nq1 * nq2 + q * nq2 + r] += rqr[n * nq1 * nq2 + q * nq2 + r] * db;
                    }
                }
            }
        }

        // 9b: D1^T rqs
        for(unsigned int p = 0; p < nq0; ++p){
            for(unsigned int n = 0; n < nq1; ++n){
                for(unsigned int q = 0; q < nq1; ++q){
                    const T db = dbasis1[q * nq1 + n];
                    for(unsigned int r = 0; r < nq2; ++r){
                        wsp1[p * nq1 * nq2 + q * nq2 + r] += rqs[p * nq1 * nq2 + n * nq2 + r] * db;
                    }
                }
            }
        }

        // 9c: D2^T rqt
        for(unsigned int p = 0; p < nq0; ++p){
            for(unsigned int q = 0; q < nq1; ++q){
                for(unsigned int n = 0; n < nq2; ++n){
                    const T rqt_val = rqt[p * nq1 * nq2 + q * nq2 + n];
                    for(unsigned int r = 0; r < nq2; ++r){
                        wsp1[p * nq1 * nq2 + q * nq2 + r] += rqt_val * dbasis2[r * nq2 + n];
                    }
                }
            }
        }

        // ----------------------------------------------------------
        // Interpolate to GLL nodes
        // ----------------------------------------------------------

        // step-10 : direction 2
        for(unsigned int p = 0; p < nq0; p++){
            for(unsigned int q = 0; q < nq1; q++){
                // r = 0
                for(unsigned int k = 0; k < nm2; k++){
                    wsp0[p * nq1 * nm2 + q * nm2 + k] = wsp1[p * nq1 * nq2 + q * nq2 + 0] * basis2[k * nq2 + 0];
                }
                // r > 0
                for(unsigned int r = 1; r < nq2; r++){
                    const T w_val = wsp1[p * nq1 * nq2 + q * nq2 + r];
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp0[p * nq1 * nm2 + q * nm2 + k] += w_val * basis2[k * nq2 + r];
                    }
                }
            }
        }

        // step-11 : direction 1
        for(unsigned int p = 0; p < nq0; p++){
            // q = 0
            {
                for(unsigned int j = 0; j < nm1; j++){
                    const T b = basis1[j * nq1 + 0];
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] = wsp0[p * nq1 * nm2 + 0 * nm2 + k] * b;
                    }
                }
            }
            // q > 0
            for(unsigned int q = 1; q < nq1; q++){
                for(unsigned int j = 0; j < nm1; j++){
                    const T b = basis1[j * nq1 + q];
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[p * nq1 * nm2 + q * nm2 + k] * b;
                    }
                }
            }
        }

        // step-12 : direction 0
        // p = 0
        for(unsigned int i = 0; i < nm0; i++){
            const T b = basis0[i * nq0 + 0];
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    wsp0[i * nm1 * nm2 + j * nm2 + k] = wsp1[0 * nm1 * nm2 + j * nm2 + k] * b;
                }
            }
        }
        // p > 0
        for(unsigned int p = 1; p < nq0; p++){
            for(unsigned int i = 0; i < nm0; i++){
                const T b = basis0[i * nq0 + p];
                for(unsigned int j = 0; j < nm1; j++){
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp0[i * nm1 * nm2 + j * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * b;
                    }
                }
            }
        }

        // step-13 : Copy from wsp0 to out
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    out[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] =
                        wsp0[i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }
    }

    // Return element-wise square of out array and apply sum reduction
    return std::transform_reduce(
        out,
        out + nelmt * nm0 * nm1 * nm2,
        out,
        T{},
        [](T lhs, T rhs){ return rhs + lhs; },
        [](T val1, T val2){ return val1 * val2; });
}

}  //namespace Serial
}  //namespace BK3

#endif //BK3_SERIALKERNELS_OTF_HPP