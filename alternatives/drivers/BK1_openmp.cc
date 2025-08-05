#include <iostream>
#include <fstream>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cmath>
#include <array>
#include <tuple>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "timer.hpp"
#include "common.hpp"

template<typename T, int nq0, int nq1, int nq2, typename index_type = int>
void SumFactorization(
    const size_t nelmt, 
    const T *__restrict__ basis0, 
    const T *__restrict__ basis1,
    const T *__restrict__ basis2, 
    const T *__restrict__ JxW, 
    const T *__restrict__ in, 
          T *__restrict__ out)
{

    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    #pragma omp target \
        map(to: basis0[:nm0*nq0], basis1[:nm1*nq1], basis2[:nm2*nq2], in[:nelmt*nm0*nm1*nm2], JxW[:nelmt*nq0*nq1*nq2]) \
        map(from: out[:nelmt*nm0*nm1*nm2])
    #pragma omp teams loop
    for(size_t e = 0; e < nelmt; ++e) {

        T wsp0[nq0*nq1*nq2];
        T wsp1[nq0*nq1*nq2];

        //step-1 : Copy from in to the wsp0
        for(index_type i = 0; i < nm0; ++i){
            for(index_type j = 0; j < nm1; ++j){
                for(index_type k = 0; k < nm2; ++k){
                    wsp0[i * nm1 * nm2 + j * nm2 + k] = in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }

        //step-2 : direction 0
        for(index_type p = 0; p < nq0; ++p){
            for(index_type k = 0; k < nm2; ++k){
                for(index_type j = 0; j < nm1; ++j){
                    T tmp{0};
                    for(index_type i = 0; i < nm0; ++i){
                        tmp += wsp0[i * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
                    }
                    wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                }
            }
        }

        //step-3 : direction 1
        for(index_type q = 0; q < nq1; ++q){
            for(index_type p = 0; p < nq0; ++p){
                for(index_type k = 0; k < nm2; ++k){
                    T tmp{0};
                    for(index_type j = 0; j < nm1; ++j){
                        tmp += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis1[q * nm1 + j];
                    }
                    wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                }
            }
        }

        //step-4 : direction 2
        for(index_type r = 0; r < nq2; ++r){
            for(index_type q = 0; q < nq1; ++q){
                for(index_type p = 0; p < nq0; ++p){
                    T tmp{0};
                    for(index_type k = 0; k < nm2; ++k){
                        tmp += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis2[r * nm2 + k];
                    }
                    wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
                }
            }
        }

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(index_type r = 0; r < nq2; ++r){
            for(index_type q = 0; q < nq1; ++q){
                for(index_type p = 0; p < nq0; ++p){
                    wsp1[p * nq1 * nq2 + q * nq2 + r] *= JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
                }
            }
        }

        //step-6 : direction 2
        for(index_type k = 0; k < nm2; ++k){
            for(index_type q = 0; q < nq1; ++q){
                for(index_type p = 0; p < nq0; ++p){
                    T tmp{0};
                    for(index_type r = 0; r < nq2; ++r){
                        tmp += wsp1[p * nq1 * nq2 + q * nq2 + r] * basis2[r * nm2 + k];
                    }
                    wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                }
            }
        }

        //step-7 : direction 1
        for(index_type j = 0; j < nm1; ++j){
            for(index_type k = 0; k < nm2; ++k){
                for(index_type p = 0; p < nq0; ++p){
                    T tmp{0};
                    for(index_type q = 0; q < nq1; ++q){
                        tmp += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis1[q * nm1 + j];
                    }
                    wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                }
            }
        }

        //step-8 : direction 0
        for(index_type i = 0; i < nm0; ++i){
            for(index_type j = 0; j < nm1; ++j){
                for(index_type k = 0; k < nm2; ++k){
                    T tmp{0};
                    for(index_type p = 0; p < nq0; ++p){
                        tmp += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
                    }
                    wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
                }
            }
        }

        //step-9 : Copy from wsp3 to out
        for(index_type i = 0; i < nm0; ++i){
            for(index_type j = 0; j < nm1; ++j){
                for(index_type k = 0; k < nm2; ++k){
                    out[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] = wsp0[i * nm1 * nm2 + j * nm2 + k];
                }
            }
        }

    } // end loop over elements

}



template<typename T, int nq0, int nq1, int nq2, typename index_type = int>
void SumFactorizationNested(
    const size_t nelmt, 
    const T *__restrict__ basis0, 
    const T *__restrict__ basis1,
    const T *__restrict__ basis2, 
    const T *__restrict__ JxW, 
    const T *__restrict__ in, 
          T *__restrict__ out)
{

    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

#define fill_zero(wsp) std::fill(wsp,wsp+nq0*nq1*nq2,(T)0.0)

#ifdef _OPENMP
    const int max_teams = omp_get_max_teams();
#else
    const int max_teams = 1;
#endif
    const size_t chunksz = (nelmt + max_teams - 1)/max_teams;

    #pragma omp target \
        map(to: basis0[:nm0*nq0], basis1[:nm1*nq1], basis2[:nm2*nq2]) \
        map(to: in[:nelmt*nm0*nm1*nm2], JxW[:nelmt*nq0*nq1*nq2]) \
        map(from: out[:nelmt*nm0*nm1*nm2])
    #pragma omp teams distribute num_teams(max_teams)
    for(int cidx = 0; cidx < max_teams; ++cidx) {

        const size_t lo = cidx*chunksz;
        const size_t hi = std::min((cidx+1)*chunksz,nelmt);

        T sbasis0[nm0*nq0];
        T sbasis1[nm1*nq1];
        T sbasis2[nm2*nq2];

        std::copy(basis0,basis0+nm0*nq0,sbasis0);
        std::copy(basis1,basis1+nm1*nq1,sbasis1);
        std::copy(basis2,basis2+nm2*nq2,sbasis2);

#if defined(__GNUC__)
        #pragma omp loop
#else
        #pragma omp loop bind(parallel)
#endif
        for(size_t e = lo; e < hi; ++e) {

            T wsp0[nq0*nq1*nq2] = {};
            T wsp1[nq0*nq1*nq2] = {};

            //step-1 : Copy from in to the wsp0
            for(index_type i = 0; i < nm0; ++i){
                for(index_type j = 0; j < nm1; ++j){
                    for(index_type k = 0; k < nm2; ++k){
                        wsp0[i * nm1 * nm2 + j * nm2 + k] = in[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k];
                    }
                }
            }

            //step-2 : direction 0
            for(index_type p = 0; p < nq0; ++p){
                for(index_type k = 0; k < nm2; ++k){
                    for(index_type j = 0; j < nm1; ++j){
                        for(index_type i = 0; i < nm0; ++i){
                            wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[i * nm1 * nm2 + j * nm2 + k] * sbasis0[p * nm0 + i];
                        }
                    }
                }
            }

            fill_zero(wsp0);

            //step-3 : direction 1
            for(index_type q = 0; q < nq1; ++q){
                for(index_type p = 0; p < nq0; ++p){
                    for(index_type k = 0; k < nm2; ++k){
                        for(index_type j = 0; j < nm1; ++j){
                            wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * sbasis1[q * nm1 + j];
                        }
                    }
                }
            }

            fill_zero(wsp1);

            //step-4 : direction 2
            for(index_type r = 0; r < nq2; ++r){
                for(index_type q = 0; q < nq1; ++q){
                    for(index_type p = 0; p < nq0; ++p){
                        for(index_type k = 0; k < nm2; ++k){
                            wsp1[p * nq1 * nq2 + q * nq2 + r] += wsp0[q * nq0 * nm2 + p * nm2 + k] * sbasis2[r * nm2 + k];
                        }
                    }
                }
            }

            fill_zero(wsp0);

            //Reverse Operations

            //step-5 : Multiply with weights and determinant of Jacobi
            for(index_type r = 0; r < nq2; ++r){
                for(index_type q = 0; q < nq1; ++q){
                    for(index_type p = 0; p < nq0; ++p){
                        wsp1[p * nq1 * nq2 + q * nq2 + r] *= JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
                    }
                }
            }


            //step-6 : direction 2
            for(index_type k = 0; k < nm2; ++k){
                for(index_type q = 0; q < nq1; ++q){
                    for(index_type p = 0; p < nq0; ++p){
                        for(index_type r = 0; r < nq2; ++r){
                            wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nq1 * nq2 + q * nq2 + r] * sbasis2[r * nm2 + k];
                        }
                    }
                }
            }

            fill_zero(wsp1);

            //step-7 : direction 1
            for(index_type j = 0; j < nm1; ++j){
                for(index_type k = 0; k < nm2; ++k){
                    for(index_type p = 0; p < nq0; ++p){
                        for(index_type q = 0; q < nq1; ++q){
                            wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[q * nq0 * nm2 + p * nm2 + k] * sbasis1[q * nm1 + j];
                        }
                    }
                }
            }

            fill_zero(wsp0);

            //step-8 : direction 0
            for(index_type i = 0; i < nm0; ++i){
                for(index_type j = 0; j < nm1; ++j){
                    for(index_type k = 0; k < nm2; ++k){
                        for(index_type p = 0; p < nq0; ++p){
                            wsp0[i * nm1 * nm2 + j * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * sbasis0[p * nm0 + i];
                        }
                    }
                }
            }

            //step-9 : Copy from wsp3 to out
            for(index_type i = 0; i < nm0; ++i){
                for(index_type j = 0; j < nm1; ++j){
                    for(index_type k = 0; k < nm2; ++k){
                        out[e * nm0 * nm1 * nm2 + i * nm1 * nm2 + j * nm2 + k] = wsp0[i * nm1 * nm2 + j * nm2 + k];
                    }
                }
            }

        } // end loop over elements

    } // end loop over chunks

}




template<typename T, int nq0, int nq1, int nq2, typename index_type = int>
void BwdTransHexKernel_QP_1D(
    int nelmt,
    const T *__restrict__ d_basis0,
    const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2,
    const T *__restrict__ JxW,
    const T *__restrict__ d_in,
          T *__restrict__ d_out)
{

    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    // Amount of shared memory needed
    constexpr int MAX_SHARED = 2 * (nq0*nq1*nq2) + nq0*nm0 + nq1*nm1 + nq2*nm2;

#pragma omp target map(to: d_basis0[0:nq0*nm0],d_basis1[0:nq1*nm1],d_basis2[0:nq2*nm2]) \
    map(to: JxW[0:nelmt * nq0 * nq1 * nq2]) \
    map(to: d_in[0:nelmt * nm0 * nm1 * nm2]) \
    map(from: d_out[0:nelmt * nm0 * nm1 * nm2])
#pragma omp teams
{
    T shared[MAX_SHARED];
    T *__restrict__ s_basis0 = shared;
    T *__restrict__ s_basis1 = s_basis0 + nm0 * nq0;
    T *__restrict__ s_basis2 = s_basis1 + nm1 * nq1;
    T *__restrict__ s_wsp0 = s_basis2 + nm2 * nq2;
    T *__restrict__ s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type i = 0; i < nq0 * nm0; ++i){
        s_basis0[i] = d_basis0[i];
    }

    for(index_type j = 0; j < nq1 * nm1; ++j) {
        s_basis1[j] = d_basis1[j];
    }

    for(index_type k = 0; k < nq2 * nm2; ++k){
        s_basis2[k] = d_basis2[k];
    }

#pragma omp parallel
{
#ifndef _OPENMP
    const int me = 0;
    const int nthreads = 1;
    const int nblk = 1;
#else
    const int me = omp_get_thread_num(); // threadIdx.x
    const int nthreads = omp_get_num_threads(); // blockDim.x
    const int nblk = omp_get_num_teams(); // gridDim.x
#endif


#ifndef _OPENMP
    int e = 0;
#else
    int e = omp_get_team_num(); // blockIdx.x;
#endif

    //std::cout << "nelmt = " << nelmt << '\n';

    while(e < nelmt)
    {
        //std::cout << "updating element " << e << '/' << nelmt << '\n';
        //std::println("{} {} {} {} {} {}",i,j,k,p,q,r);

        //step-1 : Copy from in to the wsp0
        for(index_type tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        #pragma omp barrier


        //step-2 : direction 0
        for(index_type tid = me; tid < nq0 * nm1 * nm2; tid += nthreads)
        {
            index_type p = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-3 : direction 1
        for(index_type tid = me; tid < nq0 * nq1 * nm2; tid += nthreads)
        {
            index_type q = tid / (nq0 * nm2);
            index_type p = (tid % (nq0 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        #pragma omp barrier



        //step-4 : direction 2
        for(index_type tid = me; tid < nq0 * nq1 * nq2; tid += nthreads)
        {
            index_type p = tid / (nq1 * nq2);
            index_type q = (tid % (nq1 * nq2)) / nq2;
            index_type r = tid % nq2;

            T tmp = 0.0;
            for(index_type k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        #pragma omp barrier

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(index_type tid = me; tid < nq0 * nq1 * nq2; tid += nthreads){
            s_wsp1[tid] *= JxW[e * nq0 * nq1 * nq2 + tid];
        }
        #pragma omp barrier



        //step-6 : direction 2
        for(index_type tid = me; tid < nq0 * nq1 * nm2; tid += nthreads)
        {
            index_type q = tid / (nq0 * nm2);
            index_type p = (tid % (nq0 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-7 : direction 1
        for(index_type tid = me; tid < nm1 * nm2 * nq0; tid += nthreads)
        {
            index_type p = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-8 : direction 0
        for(index_type tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            index_type i = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier

        //step-9 : Copy wsp0 to out
        for(index_type tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        #pragma omp barrier

        e += nblk; // gridDim.x

    } // while

} // omp parallel
} // omp teams

}



template<typename T, int nq0, int nq1, int nq2, typename index_type = int>
void BwdTransHexKernel_QP_1D_hybrid(
    int nelmt,
    const T *__restrict__ d_basis0,
    const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2,
    const T *__restrict__ JxW,
    const T *__restrict__ d_in,
          T *__restrict__ d_out)
{

    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    // Amount of shared memory needed
    constexpr int MAX_SHARED = 2 * (nq0*nq1*nq2) + nq0*nm0 + nq1*nm1 + nq2*nm2;

#pragma omp target map(to: d_basis0[0:nq0*nm0],d_basis1[0:nq1*nm1],d_basis2[0:nq2*nm2]) \
    map(to: JxW[0:nelmt * nq0 * nq1 * nq2]) \
    map(to: d_in[0:nelmt * nm0 * nm1 * nm2]) \
    map(from: d_out[0:nelmt * nm0 * nm1 * nm2])
#pragma omp teams
{
    T shared[MAX_SHARED];
    T *__restrict__ s_basis0 = shared;
    T *__restrict__ s_basis1 = s_basis0 + nm0 * nq0;
    T *__restrict__ s_basis2 = s_basis1 + nm1 * nq1;
    T *__restrict__ s_wsp0 = s_basis2 + nm2 * nq2;
    T *__restrict__ s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //copy to shared memory
    std::copy(d_basis0,d_basis0+nq0*nm0,s_basis0);
    std::copy(d_basis1,d_basis1+nq1*nm1,s_basis1);
    std::copy(d_basis2,d_basis2+nq2*nm2,s_basis2);

#ifndef _OPENMP
    const int nblk = 1;
    int e = 0;
#else
    const int nblk = omp_get_num_teams(); // gridDim.x
    int e = omp_get_team_num(); // blockIdx.x;
#endif

    while(e < nelmt) {

        #pragma omp parallel firstprivate(e)
        {

        //std::cout << "updating element " << e << '/' << nelmt << '\n';
        //std::println("{} {} {} {} {} {}",i,j,k,p,q,r);

        //step-1 : Copy from in to the wsp0
        #pragma omp for
        for(index_type tid = 0; tid < nm0 * nm1 * nm2; ++tid)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }


        //step-2 : direction 0
        #pragma omp for
        for(index_type tid = 0; tid < nq0 * nm1 * nm2; ++tid)
        {
            index_type p = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }


        //step-3 : direction 1
        #pragma omp for
        for(index_type tid = 0; tid < nq0 * nq1 * nm2; ++tid)
        {
            index_type q = tid / (nq0 * nm2);
            index_type p = (tid % (nq0 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }


        //step-4 : direction 2
        #pragma omp for
        for(index_type tid = 0; tid < nq0 * nq1 * nq2; ++tid)
        {
            index_type p = tid / (nq1 * nq2);
            index_type q = (tid % (nq1 * nq2)) / nq2;
            index_type r = tid % nq2;

            T tmp = 0.0;
            for(index_type k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        #pragma omp for
        for(index_type tid = 0; tid < nq0 * nq1 * nq2; ++tid){
            s_wsp1[tid] *= JxW[e * nq0 * nq1 * nq2 + tid];
        }


        //step-6 : direction 2
        #pragma omp for
        for(index_type tid = 0; tid < nq0 * nq1 * nm2; ++tid)
        {
            index_type q = tid / (nq0 * nm2);
            index_type p = (tid % (nq0 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }


        //step-7 : direction 1
        #pragma omp for
        for(index_type tid = 0; tid < nm1 * nm2 * nq0; ++tid)
        {
            index_type p = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }


        //step-8 : direction 0
        #pragma omp for
        for(index_type tid = 0; tid < nm0 * nm1 * nm2; ++tid)
        {
            index_type i = tid / (nm1 * nm2);
            index_type j = (tid % (nm1 * nm2)) / nm2;
            index_type k = tid % nm2;

            T tmp = 0.0;
            for(index_type p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }

        //step-9 : Copy wsp0 to out
        #pragma omp for
        for(index_type tid = 0; tid < nm0 * nm1 * nm2; ++tid)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }

        } // parallel for

        e += nblk; // gridDim.x

    } // while

} // omp teams

}


extern "C"
void sevalf(
    int nm0, int nm1, int nm2,
    int nq0, int nq1, int nq2,
    int nelmt,
    const float* __restrict__ basis0, 
    const float* __restrict__ basis1,
    const float* __restrict__ basis2,
    const float* __restrict__ JxW,
    const float* __restrict__ in,
          float* __restrict__ out);


template<typename T, int nq0, int nq1, int nq2>
void run_test(const size_t nelmt, const int ntests, bool show_norm)
{

    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

    //Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, (T)0.0);

    //Initialization of basis functions
    for(int p = 0; p < nq0; p++) {
        for(int i = 0; i < nm0; i++) {
            basis0[p * nm0 + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for(int q = 0; q < nq1; q++) {
        for(int j = 0; j < nm1; j++) {
            basis1[q * nm1 + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for(int r = 0; r < nq2; r++) {
        for(int k = 0; k < nm2; k++) {
            basis2[r * nm2 + k] = std::cos((T)(r * nm2 + k));
        }
    }

    [[maybe_unused]] const size_t size_JxW = nelmt * nq0 * nq1 * nq2;
    [[maybe_unused]] const size_t size_in = nelmt * nm0 * nm1 * nm2;
    [[maybe_unused]] const size_t size_out = nelmt * nm0 * nm1 * nm2;
    [[maybe_unused]] const size_t size_basis0 = nq0*nm0;
    [[maybe_unused]] const size_t size_basis1 = nq1*nm1;
    [[maybe_unused]] const size_t size_basis2 = nq2*nm2;

    T *d_JxW    = JxW.data();
    T *d_in     = in.data();
    T *d_out    = out.data();
    T *d_basis0 = basis0.data();
    T *d_basis1 = basis1.data();
    T *d_basis2 = basis2.data();

    double elapsed = std::numeric_limits<double>::max();
    Timer testTimer;

    #pragma omp target data \
            map(to: d_basis0[:size_basis0], d_basis1[:size_basis1], d_basis2[:size_basis2]) \
            map(to: d_in[:size_in], d_JxW[:size_JxW]) \
            map(tofrom: d_out[:size_out])
    for (int t = 0; t < ntests; ++t)
    {
        testTimer.start();
#if 1
//        sevalf(nm0,nm1,nm2,nq0,nq1,nq2,nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
        SumFactorization<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
//        SumFactorizationNested<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
#else
//        BwdTransHexKernel_QP_1D<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
        BwdTransHexKernel_QP_1D_hybrid<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
#endif
        testTimer.stop();
        elapsed = std::min(elapsed, testTimer.elapsedSeconds());
    }

    const size_t ndofs = nelmt * nm0 * nm1 * nm2;

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * ndofs / elapsed;
    };

    std::cout << "SumFactorization -> " << "nelmt = " << nelmt <<" GDoF/s = " << dof_rate(elapsed) << std::endl;

    if (show_norm) {
        double normSqr = squared_norm<T,double>(out.data(),out.size());
        std::cout << "# OpenMP kernel norm = " << std::sqrt(normSqr) << '\n';
    }
}

int main(int argc, char **argv){

    int nq0       = (argc > 1) ? atoi(argv[1]) : 4;
    int nq1       = (argc > 2) ? atoi(argv[2]) : nq0;
    int nq2       = (argc > 3) ? atoi(argv[3]) : nq0;
    size_t nelmt  = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    int ntests    = (argc > 5) ? atoi(argv[5]) : 5;

    const char *env = getenv("SHOW_NORM");
    bool show_norm = (env && strcmp(env, "1") == 0);

    std::cout.precision(8);

// Note: adding more cases can increase the compilation time.

    if (nq0 == nq1 && nq1 == nq2) {
        switch (nq0) {
            case 2: run_test<float,2,2,2>(nelmt, ntests, show_norm); break;
            case 3: run_test<float,3,3,3>(nelmt, ntests, show_norm); break;
            case 4: run_test<float,4,4,4>(nelmt, ntests, show_norm); break;
            case 5: run_test<float,5,5,5>(nelmt, ntests, show_norm); break;
//            case 6: run_test<float,6,6,6>(nelmt, ntests, show_norm); break;
//            case 7: run_test<float,7,7,7>(nelmt, ntests, show_norm); break;
//            case 8: run_test<float,8,8,8>(nelmt, ntests, show_norm); break;
            default: return unsupported(nq0, nq1, nq2);
        }
    } else {
        // Mixed cases aren't supported
        return unsupported(nq0, nq1, nq2);
    }

    return 0;
}
