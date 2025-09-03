#ifndef BK3_SERIALKERNELS_HPP
#define BK3_SERIALKERNELS_HPP

#include <numeric>

namespace BK3{
namespace Serial{

template<typename T>
T SumFactorization( const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ basis0, const T *__restrict__ basis1, const T *__restrict__ basis2, 
    const T *__restrict__ dbasis0, const T *__restrict__ dbasis1, const T *__restrict__ dbasis2, 
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    // Intermediate vals
    T *wsp0 = new T[nq0 * nq1 * nq2];
    T *wsp1 = new T[nq0 * nq1 * nq2];
    T *rqr = new T[nelmt * nq0 * nq1 * nq2];
    T *rqs = new T[nelmt * nq0 * nq1 * nq2];
    T *rqt = new T[nelmt * nq0 * nq1 * nq2];

    /*
    Interpolate to GL nodes
    */
    for(unsigned int e = 0; e < nelmt; ++e){
        
        std::fill(wsp0, wsp0 + nq0 * nq1 * nq2, (T)0);
        std::fill(wsp1, wsp1 + nq0 * nq1 * nq2, (T)0);

        //step-1 : Copy from in to the wsp0
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    wsp0[i * nm1 * nm2 + j * nm2 + k] = in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }

        //step-2 : direction 0
        for(unsigned int p = 0; p < nq0; p++){
            for(unsigned int k = 0; k < nm2; k++){
                for(unsigned int j = 0; j < nm1; j++){
                    for(unsigned int i = 0; i < nm0; i++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[i * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
                    }
                }
            }
        }
        std::fill(wsp0, wsp0 + nq0 * nq1 * nq2, (T)0);

        //step-3 : direction 1
        for(unsigned int q = 0; q < nq1; q++){
            for(unsigned int p = 0; p < nq0; p++){
                for(unsigned int k = 0; k < nm2; k++){
                    for(unsigned int j = 0; j < nm1; j++){
                        wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis1[q * nm1 + j];
                    }
                }
            }
        }
        std::fill(wsp1, wsp1 + nq0 * nq1 * nq2, (T)0);


        //step-4 : direction 2
        for(unsigned int r = 0; r < nq2; r++){
            for(unsigned int q = 0; q < nq1; q++){
                for(unsigned int p = 0; p < nq0; p++){
                    for(unsigned int k = 0; k < nm2; k++){
                        wsp1[p * nq1 * nq2 + q * nq2 + r] += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis2[r * nm2 + k];
                    }
                }
            }
        }
        
        // Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;

        for(unsigned int p = 0; p < nq0; ++p){
            for(unsigned int q = 0; q < nq1; ++q){              
                for(unsigned int r = 0; r < nq2; ++r){

                    //step-5 : Load Geometric Factors, coalesced access
                    Grr = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 0 * nq2 + r];
                    Grs = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 1 * nq2 + r];
                    Grt = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 2 * nq2 + r];
                    Gss = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 3 * nq2 + r];
                    Gst = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 4 * nq2 + r];
                    Gtt = G[e * nq0 * nq1 * 6 * nq2 + p * nq1 * 6 * nq2 + q * 6 * nq2 + 5 * nq2 + r];
                    
                    //step-6 : Multiply by D
                    T qr = 0.0; T qs = 0.0; T qt = 0.0;

                    for(unsigned int n = 0; n < nq0; ++n){
                        qr += wsp1[n * nq1 * nq2 + q * nq2 + r] * dbasis0[p * nq0 + n];
                    }

                    for(unsigned int n = 0; n < nq1; ++n){
                        qs += wsp1[p * nq1 * nq2 + n * nq2 + r] * dbasis1[q * nq1 + n];
                    }

                    for(unsigned int n = 0; n < nq2; ++n){
                        qt += wsp1[p * nq1 * nq2 + q * nq2 + n] * dbasis2[r * nq2 + n];
                    }

                    // step-7 : Apply chain rule
                    rqr[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = Grr * qr + Grs * qs + Grt * qt;
                    rqs[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = Grs * qr + Gss * qs + Gst * qt;
                    rqt[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        // step-8 : Compute out vector in GL nodes
        for(unsigned int p = 0; p < nq0; ++p){                      
            for(unsigned int q = 0; q < nq1; ++q){ 
                for(unsigned int r = 0; r < nq2; ++r){ 

                T tmp0 = (T)0;
                for(unsigned int n = 0; n < nq0; ++n)
                    tmp0 += rqr[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + q * nq2 + r] * dbasis0[n * nq0 + p];

                for(unsigned int n = 0; n < nq1; ++n)                
                    tmp0 += rqs[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + n * nq2 + r] * dbasis1[n * nq1 + q];

                for(unsigned int n = 0; n < nq2; ++n)
                    tmp0 += rqt[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + n] * dbasis2[n * nq2 + r];

                wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp0;
                }
            }
        }


        /*
        Interpolate to GLL nodes
        */

        //step-9 : direction 2
        std::fill(wsp0, wsp0 + nq0 * nq1 * nq2, (T)0);

        for(unsigned int k = 0; k < nm2; k++){
            for(unsigned int q = 0; q < nq1; q++){
                for(unsigned int p = 0; p < nq0; p++){ 
                    for(unsigned int r = 0; r < nq2; r++){
                        wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nq1 * nq2 + q * nq2 + r] * basis2[r * nm2 + k];
                    }
                }
            }
        }
        std::fill(wsp1, wsp1 + nq0 * nq1 * nq2, (T)0);

        //step-10 : direction 1
        for(unsigned int j = 0; j < nm1; j++){
            for(unsigned int k = 0; k < nm2; k++){
                for(unsigned int p = 0; p < nq0; p++){
                    for(unsigned int q = 0; q < nq1; q++){
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis1[q * nm1 + j];
                    }
                }
            }
        }
        std::fill(wsp0, wsp0 + nq0 * nq1 * nq2, (T)0);


        //step-11 : direction 0
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    for(unsigned int p = 0; p < nq0; p++){
                        wsp0[i * nm1 * nm2 + j * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
                    }
                }
            }
        }

        //step-12 : Copy from wsp0 to out
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    out[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] = wsp0[i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }
    }
    
    //return element-wise square of in array and apply sum reduction
    return std::transform_reduce(out, out + nelmt * nm0 * nm1 * nm2,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
    
    delete[] wsp0; delete[] wsp1; delete[] rqr; delete[] rqs; delete[] rqt;
}

}  //namespace Serial
}  //namespace BK3

#endif //BK3_SERIALKERNELS_HPP
