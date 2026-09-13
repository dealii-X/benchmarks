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
    const unsigned int ncomp = 3;
    const unsigned int ndof_1D = nm * nm * nm;
    const unsigned int ndof_total = ncomp * ndof_1D;

    T *wsp0 = new T[nq * nq * nq];
    T *wsp1 = new T[nq * nq * nq];
    T *rqr  = new T[nq * nq * nq];
    T *rqs  = new T[nq * nq * nq];
    T *rqt  = new T[nq * nq * nq];

    for(unsigned int e = 0; e < nelmt; ++e){
        
        const size_t element_offset = static_cast<size_t>(e) * ndof_1D;
        const size_t component_offset = static_cast<size_t>(nelmt) * ndof_1D;

        for(unsigned int c = 0; c < ncomp; ++c){

            const T* in_c = in + c * component_offset + element_offset;
            T* out_c = out + c * component_offset + element_offset;

            std::fill(wsp0, wsp0 + nq * nq * nq, (T)0);
            std::fill(wsp1, wsp1 + nq * nq * nq, (T)0);
            std::fill(rqr, rqr + nq * nq * nq, (T)0);
            std::fill(rqs, rqs + nq * nq * nq, (T)0);
            std::fill(rqt, rqt + nq * nq * nq, (T)0);

            /*
            Interpolate to GL nodes
            */

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
            std::fill(wsp1, wsp1 + nq * nq * nq, (T)0);

            //step-4 : direction 2
            for(unsigned int r = 0; r < nq; r++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int k = 0; k < nm; k++){
                            wsp1[p * nq * nq + q * nq + r] += wsp0[q * nq * nm + p * nm + k] * basis[k * nq + r];
                        }
                    }
                }
            }
            
            // Geometric vals
            T Grr, Grs, Grt, Gss, Gst, Gtt;

            for(unsigned int p = 0; p < nq; ++p){
                for(unsigned int q = 0; q < nq; ++q){              
                    for(unsigned int r = 0; r < nq; ++r){

                        //step-5 : Load Geometric Factors
                        Grr = G[e * nq * nq * nq * 6 + 0 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Grs = G[e * nq * nq * nq * 6 + 1 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Grt = G[e * nq * nq * nq * 6 + 2 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gss = G[e * nq * nq * nq * 6 + 3 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gst = G[e * nq * nq * nq * 6 + 4 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gtt = G[e * nq * nq * nq * 6 + 5 * nq * nq * nq + p * nq * nq + q * nq + r];
                        
                        //step-6 : Multiply by quadrature-space D
                        T qr = 0.0; T qs = 0.0; T qt = 0.0;

                        for(unsigned int n = 0; n < nq; ++n){
                            qr += wsp1[n * nq * nq + q * nq + r] * dbasis[n * nq + p];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            qs += wsp1[p * nq * nq + n * nq + r] * dbasis[n * nq + q];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            qt += wsp1[p * nq * nq + q * nq + n] * dbasis[n * nq + r];
                        }

                        // step-7 : Apply chain rule
                        rqr[p * nq * nq + q * nq + r] = Grr * qt + Grs * qs + Grt * qr;
                        rqs[p * nq * nq + q * nq + r] = Grs * qt + Gss * qs + Gst * qr;
                        rqt[p * nq * nq + q * nq + r] = Grt * qt + Gst * qs + Gtt * qr;
                    }
                }
            }

            // step-8 : Compute out vector in GL nodes
            for(unsigned int p = 0; p < nq; ++p){                      
                for(unsigned int q = 0; q < nq; ++q){ 
                    for(unsigned int r = 0; r < nq; ++r){ 

                        T tmp0 = (T)0;
                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += rqr[n * nq * nq + q * nq + r] * dbasis[p * nq + n];

                        for(unsigned int n = 0; n < nq; ++n)                
                            tmp0 += rqs[p * nq * nq + n * nq + r] * dbasis[q * nq + n];

                        for(unsigned int n = 0; n < nq; ++n)
                            tmp0 += rqt[p * nq * nq + q * nq + n] * dbasis[r * nq + n];

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

    delete[] wsp0; delete[] wsp1; delete[] rqr; delete[] rqs; delete[] rqt;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
}

#endif   //BK4_SERIALKERNELS_HPP