#include <iostream>
#include <fstream>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cmath>
#include <array>
#include <tuple>
#include <cstring>

#include "timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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

#define fill_zero(wsp) std::fill(wsp,wsp+nq0*nq1*nq2,(T)0.0)


//    #pragma omp parallel for default(shared) firstprivate(wsp0,wsp1)
    #pragma omp target \
        map(to: basis0[:nm0*nq0], basis1[:nm1*nq1], basis2[:nm2*nq2], in[:nelmt*nm0*nm1*nm2], JxW[:nelmt*nq0*nq1*nq2]) \
        map(from: out[:nelmt*nm0*nm1*nm2])
    #pragma omp teams loop
    for(size_t e = 0; e < nelmt; ++e) {

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
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[i * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
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
                        wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis1[q * nm1 + j];
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
                        wsp1[p * nq1 * nq2 + q * nq2 + r] += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis2[r * nm2 + k];
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
                        wsp0[q * nq0 * nm2 + p * nm2 + k] += wsp1[p * nq1 * nq2 + q * nq2 + r] * basis2[r * nm2 + k];
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
                        wsp1[p * nm1 * nm2 + j * nm2 + k] += wsp0[q * nq0 * nm2 + p * nm2 + k] * basis1[q * nm1 + j];
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
                        wsp0[i * nm1 * nm2 + j * nm2 + k] += wsp1[p * nm1 * nm2 + j * nm2 + k] * basis0[p * nm0 + i];
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

}


template<typename T, int nq0, int nq1, int nq2>
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
    const int MAX_SHARED = 2 * (nq0*nq1*nq2) + nq0*nm0 + nq1*nm1 + nq2*nm2;

#pragma omp target map(to: d_basis0[0:nq0*nm0],d_basis1[0:nq1*nm1],d_basis2[0:nq2*nm2]) \
    map(to: JxW[0:nelmt * nq0 * nq1 * nq2]) \
    map(to: d_in[0:nelmt * nm0 * nm1 * nm2]) \
    map(from: d_out[0:nelmt * nm0 * nm1 * nm2])
#pragma omp teams
{
    T shared[MAX_SHARED];
    T *s_basis0 = &shared[0];
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

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

    //copy to shared memory
    for(int tid = me; tid < nq0 * nm0; tid += nthreads)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(int tid = me; tid < nq1 * nm1; tid += nthreads)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(int tid = me; tid < nq2 * nm2; tid += nthreads)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    int i, j, k, p, q, r;

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
        for(int tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        #pragma omp barrier


        //step-2 : direction 0
        for(int tid = me; tid < nq0 * nm1 * nm2; tid += nthreads)
        {
            p = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-3 : direction 1
        for(int tid = me; tid < nq0 * nq1 * nm2; tid += nthreads)
        {
            q = tid / (nq0 * nm2);
            p = (tid % (nq0 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        #pragma omp barrier



        //step-4 : direction 2
        for(int tid = me; tid < nq0 * nq1 * nq2; tid += nthreads)
        {
            p = tid / (nq1 * nq2);
            q = (tid % (nq1 * nq2)) / nq2;
            r = tid % nq2;

            T tmp = 0.0;
            for(int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        #pragma omp barrier

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(int tid = me; tid < nq0 * nq1 * nq2; tid += nthreads){
            s_wsp1[tid] *= JxW[e * nq0 * nq1 * nq2 + tid];
        }
        #pragma omp barrier



        //step-6 : direction 2
        for(int tid = me; tid < nq0 * nq1 * nm2; tid += nthreads)
        {
            q = tid / (nq0 * nm2);
            p = (tid % (nq0 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-7 : direction 1
        for(int tid = me; tid < nm1 * nm2 * nq0; tid += nthreads)
        {
            p = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier


        //step-8 : direction 0
        for(int tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            i = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        #pragma omp barrier

        //step-9 : Copy wsp0 to out
        for(int tid = me; tid < nm0 * nm1 * nm2; tid += nthreads)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        #pragma omp barrier

        e += nblk; // gridDim.x

    } // while

} // omp parallel
} // omp teams

}



// Global settings
int show_norm = -1; // -1 means uninitialized

template<typename T, int nq0, int nq1, int nq2>
void run_test(
    const size_t nelmt,
    const int ntests)
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
{
    for (int t = 0; t < ntests; ++t)
    {
        testTimer.start();

#if 1
        SumFactorization<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
#else
        BwdTransHexKernel_QP_1D<T,nq0,nq1,nq2>(nelmt,d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out);
#endif

        testTimer.stop();
        elapsed = std::min(elapsed, testTimer.elapsedSeconds());
    }
}

    const size_t ndofs = nelmt * nm0 * nm1 * nm2;

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * ndofs / elapsed;
    };


    // nelmt, ndofs, GDoF/s
    std::printf("%zu\t%zu\t%f\n", nelmt, ndofs, dof_rate(elapsed));

    if (show_norm) {

        double normSqr{0.0};
        for (auto h : out) {
            normSqr += ((double) h) * ((double) h);
        }

        std::cout << "# OpenMP kernel norm = " << std::sqrt(normSqr) << '\n';
    }

}


int main(int argc, char **argv){


    size_t nelmt           = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    int ntests             = (argc > 2) ? atoi(argv[2]) : 3;

    const char *env = getenv("SHOW_NORM");
    show_norm = (env && strcmp(env, "1") == 0) ? 1 : 0;

// Empty region to initialize runtime
#pragma omp target
{}

    run_test<float,4,4,4>(nelmt,ntests);

    return 0;
}
