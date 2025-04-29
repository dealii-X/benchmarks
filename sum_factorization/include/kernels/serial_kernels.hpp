#ifndef SERIALKERNELS_HPP
#define SERIALKERNELS_HPP

#include <numeric>

namespace Serial{
template<typename T>
T DirectEvaluation(const unsigned int nm0, const unsigned nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in, T *__restrict__ out)
{
    
    for(unsigned int e = 0; e < nelmt; ++e){                        //element iteration
        for(unsigned int p = 0; p < nq0; ++p){                      //quadrature point x direction iteration
            for(unsigned int q = 0; q < nq1; ++q){                  //quadrature point y direction iteration
                for(unsigned int r = 0; r < nq2; ++r){              //quadrature point z direction iteration
                    for(unsigned int i = 0; i < nm0; ++i){                  //shape function direction 0 iteration
                        for(unsigned int j = 0; j < nm1; ++j){              //shape function direction 1 iteration
                            for (unsigned int k = 0; k < nm2; ++k){         //shape function direction 2 iteration
                                out[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] += in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] * 
                                                                                          basis0[p * nm0 + i] * 
                                                                                          basis1[q * nm1 + j] * 
                                                                                          basis2[r * nm2 + k];
                            }
                        }
                    }
                }
            }
        }
    }
    

    //return element-wise square of out array and apply sum reduction
    return std::transform_reduce(out, out + nelmt * nq0 * nq1 * nq2,
                          out, T{},
                          [](T lhs, T rhs){return rhs + lhs;},
                          [](T val1, T val2){return val1 * val2;});
}






template<typename T>
T SumFactorization(const unsigned int nm0, const unsigned nm1, const unsigned int nm2,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nelmt, const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ in, T *__restrict__ out)
{

//Intermediate Arrays
T* wsp0 = new T[nm0 * nm1 * nm2];
T* wsp1 = new T[nm1 * nm2 * nq0];
T* wsp2 = new T[nm2 * nq0 * nq1];
T* wsp3 = new T[nq0 * nq1 * nq2];


for(unsigned int e = 0; e < nelmt; ++e){
    std::fill(wsp0, wsp0 + nm0 * nm1 * nm2, (T)0);
    std::fill(wsp1, wsp1 + nm1 * nm2 * nq0, (T)0);
    std::fill(wsp2, wsp2 + nm2 * nq0 * nq1, (T)0);
    std::fill(wsp3, wsp3 + nq0 * nq1 * nq2, (T)0);


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


    //step-3 : direction 1
    for(unsigned int q = 0; q < nq1; q++){
        for(unsigned int p = 0; p < nq0; p++){
            for(unsigned int k = 0; k < nm2; k++){
                for(unsigned int j = 0; j < nm1; j++){
                    wsp2[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis1[q * nm1 + j];
                }
            }
        }
    }


    //step-4 : direction 2
    for(unsigned int r = 0; r < nq2; r++){
        for(unsigned int q = 0; q < nq1; q++){
            for(unsigned int p = 0; p < nq0; p++){
                for(unsigned int k = 0; k < nm2; k++){
                    wsp3[p * nq1 * nq2 + q * nq2 + r] += wsp2[q * nq0 * nm2 + p * nm2 + k] * basis2[r * nm2 + k];
                }
            }
        }
    }


    //step-5 : Copy from wsp3 to out
    for(unsigned int r = 0; r < nq2; ++r){
        for(unsigned int q = 0; q < nq1; ++q){
            for(unsigned int p = 0; p < nq0; ++p){
                out[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = wsp3[p * nq1 * nq2 + q * nq2 + r];
            }
        }
    }
}

    
//return element-wise square of out array and apply sum reduction
return std::transform_reduce(out, out + nelmt * nq0 * nq1 * nq2,
    out, T{},
    [](T lhs, T rhs){return rhs + lhs;},
    [](T val1, T val2){return val1 * val2;});
}

} //namespace Serial


#endif //SERIALKERNELS_HPP