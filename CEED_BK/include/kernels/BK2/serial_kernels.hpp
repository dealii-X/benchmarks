#ifndef BK2_SERIALKERNELS_HPP
#define BK2_SERIALKERNELS_HPP

#include <numeric>
#include <vector>

namespace BK2{
namespace Serial{

template<typename T>
T SumFactorization(const unsigned int nq,
    const unsigned int nelmt, const T *__restrict__ basis0, const T *__restrict__ basis1,
    const T *__restrict__ basis2, const T *__restrict__ JxW, T *__restrict__ in, T *__restrict__ out)
{
    const unsigned int nm = nq - 1;

    const unsigned int ncomp = 3;
    const unsigned int ndof_1D = nm * nm * nm;
    const unsigned int ndof_total = ncomp * ndof_1D;

    //Intermediate Arrays
    std::vector<T> wsp0(nq * nq * nq);
    std::vector<T> wsp1(nq * nq * nq);

    for(unsigned int e = 0; e < nelmt; ++e){

        const size_t element_offset = static_cast<size_t>(e) * ndof_1D;
        const size_t component_offset = static_cast<size_t>(nelmt) * ndof_1D;

        for(unsigned int c = 0; c < ncomp; ++c){

            const T* in_c = in + c * component_offset + element_offset;
            T* out_c = out + c * component_offset + element_offset;

            std::fill(wsp0.begin(), wsp0.end(), T(0.0));
            std::fill(wsp1.begin(), wsp1.end(), T(0.0));

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
                            wsp1[p * nm * nm + j * nm + k] += wsp0[i * nm * nm + j * nm + k] * basis0[i * nq + p];
                        }
                    }
                }
            }
            std::fill(wsp0.begin(), wsp0.end(), T(0.0));

            //step-3 : direction 1
            for(unsigned int q = 0; q < nq; q++){
                for(unsigned int p = 0; p < nq; p++){
                    for(unsigned int k = 0; k < nm; k++){
                        for(unsigned int j = 0; j < nm; j++){
                            wsp0[q * nq * nm + p * nm + k] += wsp1[p * nm * nm + j * nm + k] * basis1[j * nq + q];
                        }
                    }
                }
            }
            std::fill(wsp1.begin(), wsp1.end(), T(0.0));


            //step-4 : direction 2
            for(unsigned int r = 0; r < nq; r++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int k = 0; k < nm; k++){
                            wsp1[p * nq * nq + q * nq + r] += wsp0[q * nq * nm + p * nm + k] * basis2[k * nq + r];
                        }
                    }
                }
            }
            std::fill(wsp0.begin(), wsp0.end(), T(0.0));

            //Reverse Operations

            //step-5 : Multiply with weights and determinant of Jacobi
            for(unsigned int r = 0; r < nq; r++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){
                        wsp1[p * nq * nq + q * nq + r] *= JxW[e * nq * nq * nq + p * nq * nq + q * nq + r];
                    }
                }
            }

            //step-6 : direction 2
            for(unsigned int k = 0; k < nm; k++){
                for(unsigned int q = 0; q < nq; q++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int r = 0; r < nq; r++){
                            wsp0[q * nq * nm + p * nm + k] += wsp1[p * nq * nq + q * nq + r] * basis2[k * nq + r];
                        }
                    }
                }
            }
            std::fill(wsp1.begin(), wsp1.end(), T(0.0));

            //step-7 : direction 1
            for(unsigned int j = 0; j < nm; j++){
                for(unsigned int k = 0; k < nm; k++){
                    for(unsigned int p = 0; p < nq; p++){
                        for(unsigned int q = 0; q < nq; q++){
                            wsp1[p * nm * nm + j * nm + k] += wsp0[q * nq * nm + p * nm + k] * basis1[j * nq + q];
                        }
                    }
                }
            }
            std::fill(wsp0.begin(), wsp0.end(), T(0.0));


            //step-8 : direction 0
            for(unsigned int i = 0; i < nm; i++){
                for(unsigned int j = 0; j < nm; j++){
                    for(unsigned int k = 0; k < nm; k++){
                        for(unsigned int p = 0; p < nq; p++){
                            wsp0[i * nm * nm + j * nm + k] += wsp1[p * nm * nm + j * nm + k] * basis0[i * nq + p];
                        }
                    }
                }
            }

            //step-9 : Copy from wsp0 to out
            for(unsigned int i = 0; i < nm; i++){
                for(unsigned int j = 0; j < nm; j++){
                    for(unsigned int k = 0; k < nm; k++){
                        out_c[i * nm * nm + j * nm + k] = wsp0[i * nm * nm + j * nm + k];
                    }
                }
            }
        }
    }


    //return element-wise square of out array and apply sum reduction
    return std::transform_reduce(out, out + nelmt * ndof_total,
        out, T{},
        [](T lhs, T rhs){return rhs + lhs;},
        [](T val1, T val2){return val1 * val2;});
}

// Single-basis overload when basis0 == basis1 == basis2 == basis
template<typename T>
inline T SumFactorization(const unsigned int nq,
    const unsigned int nelmt, const T *__restrict__ basis,
    const T *__restrict__ JxW, T *__restrict__ in, T *__restrict__ out)
{
    return SumFactorization<T>(nq, nelmt, basis, basis, basis, JxW, in, out);
}

} //namespace Serial
} //namespace BK2

#endif //BK2_SERIALKERNELS_HPP