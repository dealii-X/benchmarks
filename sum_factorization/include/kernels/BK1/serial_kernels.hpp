#ifndef BK1_SERIALKERNELS_HPP
#define BK1_SERIALKERNELS_HPP

#include <numeric>

namespace BK1{
namespace Serial{
template<typename T>
T DirectEvaluation(const unsigned int nq0, const unsigned nq1, const unsigned int nq2,
    const unsigned int nelmt, const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ JxW, T *__restrict__ in, T *__restrict__ out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    T* intermediateVals = new T[nelmt * nq0 * nq1 * nq2];
    std::fill(intermediateVals, intermediateVals + nelmt * nq0 * nq1 * nq2, (T)0);

    for(unsigned int e = 0; e < nelmt; ++e){                        //element iteration
        for(unsigned int p = 0; p < nq0; ++p){                      //quadrature point x direction iteration
            for(unsigned int q = 0; q < nq1; ++q){                  //quadrature point y direction iteration
                for(unsigned int r = 0; r < nq2; ++r){              //quadrature point z direction iteration
                    for(unsigned int i = 0; i < nm0; ++i){                  //shape function direction 0 iteration
                        for(unsigned int j = 0; j < nm1; ++j){              //shape function direction 1 iteration
                            for (unsigned int k = 0; k < nm2; ++k){         //shape function direction 2 iteration
                                intermediateVals[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] +=  in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i] * basis1[q * nm1 + j] * basis2[r * nm2 + k];
                            }
                        }
                    }
                    intermediateVals[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] *= JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
                }
            }
        }
    }
    
    //initialize in to zero
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (T)0);

    for(unsigned int e = 0; e < nelmt; ++e){                    //element iteration
        for(unsigned int i = 0; i < nm0; ++i){                  //shape function direction 0 iteration
            for(unsigned int j = 0; j < nm1; ++j){              //shape function direction 1 iteration
                for (unsigned int k = 0; k < nm2; ++k){         //shape function direction 2 iteration
                    for(unsigned int p = 0; p < nq0; ++p){                      //quadrature point x direction iteration
                        for(unsigned int q = 0; q < nq1; ++q){                  //quadrature point y direction iteration
                            for(unsigned int r = 0; r < nq2; ++r){              //quadrature point z direction iteration
                                out[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] +=  intermediateVals[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] * basis0[p * nm0 + i] * basis1[q * nm1 + j] * basis2[r * nm2 + k];
                            }
                        }
                    }
                }
            }
        }
    }

    //return element-wise square of in array and apply sum reduction
    return std::transform_reduce(out, out + nelmt * nm0 * nm1 * nm2,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}



template<typename T>
T SumFactorization(const unsigned int nq0, const unsigned nq1, const unsigned int nq2,
    const unsigned int nelmt, const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ JxW, T *__restrict__ in, T *__restrict__ out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;
    
    //Intermediate Arrays
    T* wsp0 = new T[nq0 * nq1 * nq2];
    T* wsp1 = new T[nq0 * nq1 * nq2];

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
        std::fill(wsp0, wsp0 + nq0 * nq1 * nq2, (T)0);

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int r = 0; r < nq2; r++){
            for(unsigned int q = 0; q < nq1; q++){
                for(unsigned int p = 0; p < nq0; p++){
                    wsp1[p * nq1 * nq2 + q * nq2 + r] *= JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
                }
            }
        }
  

        //step-6 : direction 2
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

        //step-7 : direction 1
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


        //step-8 : direction 0
        for(unsigned int i = 0; i < nm0; i++){
            for(unsigned int j = 0; j < nm1; j++){
                for(unsigned int k = 0; k < nm2; k++){
                    for(unsigned int p = 0; p < nq0; p++){
                        wsp0[i * nm1 * nm2 + j * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
                    }
                }
            }
        }

        //step-9 : Copy from wsp3 to out
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
}

} //namespace Serial
} //namespace BK1

#endif //BK1_SERIALKERNELS_HPP