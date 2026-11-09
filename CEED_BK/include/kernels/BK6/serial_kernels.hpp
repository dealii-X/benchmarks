#ifndef BK6_SERIALKERNELS_HPP
#define BK6_SERIALKERNELS_HPP

#include <numeric>
#include <algorithm>

namespace BK6{
namespace Serial {

template<typename T>
T SumFactorization(const unsigned int nq, 
    const unsigned int nelmt,
    const T *__restrict__ dbasis,
    const T *__restrict__ G, T *__restrict__ in, T * __restrict__ out)
{
    const unsigned int ncomp = 3;
    const unsigned int ndof_1D = nq * nq * nq;
    const unsigned int ndof_total = ncomp * ndof_1D;

    T *rqr  = new T[nq * nq * nq];
    T *rqs  = new T[nq * nq * nq];
    T *rqt  = new T[nq * nq * nq];

    for(unsigned int e = 0; e < nelmt; ++e){
        
        const size_t element_offset = static_cast<size_t>(e) * ndof_1D;
        const size_t component_offset = static_cast<size_t>(nelmt) * ndof_1D;

        for(unsigned int c = 0; c < ncomp; ++c){

            const T* in_c = in + c * component_offset + element_offset;
            T* out_c = out + c * component_offset + element_offset;

            std::fill(rqr, rqr + nq * nq * nq, (T)0);
            std::fill(rqs, rqs + nq * nq * nq, (T)0);
            std::fill(rqt, rqt + nq * nq * nq, (T)0);

            // Geometric vals
            T Grr, Grs, Grt, Gss, Gst, Gtt;

            for(unsigned int p = 0; p < nq; ++p){
                for(unsigned int q = 0; q < nq; ++q){              
                    for(unsigned int r = 0; r < nq; ++r){

                        //step-1 : Load Geometric Factors
                        Grr = G[e * nq * nq * nq * 6 + 0 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Grs = G[e * nq * nq * nq * 6 + 1 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Grt = G[e * nq * nq * nq * 6 + 2 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gss = G[e * nq * nq * nq * 6 + 3 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gst = G[e * nq * nq * nq * 6 + 4 * nq * nq * nq + p * nq * nq + q * nq + r];
                        Gtt = G[e * nq * nq * nq * 6 + 5 * nq * nq * nq + p * nq * nq + q * nq + r];
                        
                        //step-2 : Multiply by quadrature-space D
                        T qr = 0.0; T qs = 0.0; T qt = 0.0;

                        for(unsigned int n = 0; n < nq; ++n){
                            qr += in_c[n * nq * nq + q * nq + r] * dbasis[n * nq + p];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            qs += in_c[p * nq * nq + n * nq + r] * dbasis[n * nq + q];
                        }

                        for(unsigned int n = 0; n < nq; ++n){
                            qt += in_c[p * nq * nq + q * nq + n] * dbasis[n * nq + r];
                        }

                        // step-3 : Apply chain rule
                        rqr[p * nq * nq + q * nq + r] = Grr * qt + Grs * qs + Grt * qr;
                        rqs[p * nq * nq + q * nq + r] = Grs * qt + Gss * qs + Gst * qr;
                        rqt[p * nq * nq + q * nq + r] = Grt * qt + Gst * qs + Gtt * qr;
                    }
                }
            }

            // step-4 : Compute out vector in GLL nodes
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

                        out_c[p * nq * nq + q * nq + r] = tmp0;
                    }
                }
            }
        }
    }

    delete[] rqr; delete[] rqs; delete[] rqt;
    
    return std::transform_reduce(out, out + nelmt * ndof_total,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}

}  //namespace Serial
}

#endif   //BK6_SERIALKERNELS_HPP