#ifndef BK5SERIALKERNELS_HPP
#define BK5SERIALKERNELS_HPP

#include <numeric>

namespace BK5{
namespace Serial{

template<typename T>
T SumFactorization( const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const T *__restrict__ dbasis0, const T *__restrict__ dbasis1,
    const T *__restrict__ dbasis2, const T *__restrict__ G, const T *__restrict__ in, T * __restrict__ out)
{
    // Intermediate vals
    T *rqr = new T[nelmt * nq0 * nq1 * nq2];
    T *rqs = new T[nelmt * nq0 * nq1 * nq2];
    T *rqt = new T[nelmt * nq0 * nq1 * nq2];

    for(unsigned int e = 0; e < nelmt; ++e){                    
        // Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;

        for(unsigned int i = 0; i < nq0; ++i){
            for(unsigned int j = 0; j < nq1; ++j){              
                for(unsigned int k = 0; k < nq2; ++k){

                    //Load Geometric Factors, coalesced access
                    Grr = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 0 * nq2 + k];
                    Grs = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 1 * nq2 + k];
                    Grt = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 2 * nq2 + k];
                    Gss = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 3 * nq2 + k];
                    Gst = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 4 * nq2 + k];
                    Gtt = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 5 * nq2 + k];
                    
                    // Multiply by D
                    T qr = 0.0; T qs = 0.0; T qt = 0.0;

                    for(unsigned int n = 0; n < nq0; ++n){
                        qr += in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k] * dbasis0[i * nq0 + n];
                    }

                    for(unsigned int n = 0; n < nq1; ++n){
                        qs += in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k] * dbasis1[j * nq1 + n];
                    }

                    for(unsigned int n = 0; n < nq2; ++n){
                        qt += in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n] * dbasis2[k * nq2 + n];
                    }

                    // Apply chain rule
                    rqr[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                    rqs[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                    rqt[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        for(unsigned int i = 0; i < nq0; ++i){                      
            for(unsigned int j = 0; j < nq1; ++j){ 
                for(unsigned int k = 0; k < nq2; ++k){ 

                T tmp0 = (T)0;
                for(unsigned int n = 0; n < nq0; ++n)
                    tmp0 += rqr[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k] * dbasis0[n * nq0 + i];

                for(unsigned int n = 0; n < nq1; ++n)                
                    tmp0 += rqs[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k] * dbasis1[n * nq1 + j];

                for(unsigned int n = 0; n < nq2; ++n)
                    tmp0 += rqt[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n] * dbasis2[n * nq2 + k];

                out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
                }
            }
        }
    }

    //return element-wise square of in array and apply sum reduction
    return std::transform_reduce(out, out + nelmt * nq0 * nq1 * nq2,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
    
    delete[] rqr; delete[] rqs; delete[] rqt;
}

}  //namespace Serial
}  //namespace BK5

#endif //BK5SERIALKERNELS_HPP
