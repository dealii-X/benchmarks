#ifndef BK1_TEMPLATED_CUDA_KERNELS_CUH
#define BK1_TEMPLATED_CUDA_KERNELS_CUH

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <timer.hpp>
#include <vector>

namespace BK1{
namespace Parallel{

template<typename T, unsigned int nq0, unsigned int nq1, unsigned int nq2>
__global__ void BwdTransHexKernel_QP_1D_Warp(
    unsigned int nelmt,
    const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_JxW,
    const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    unsigned int warpsPerBlock = blockDim.x / warpSize;
    unsigned int warpId        = threadIdx.x / warpSize;
    int          laneId        = threadIdx.x % warpSize;       // position of thread in warp

    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + warpsPerBlock * nq0 * nq1 * nq2;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }


    for(unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }


    for(unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }
    __syncthreads();

    unsigned int e = blockIdx.x * warpsPerBlock + warpId;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = laneId; tid < nm0 * nm1 * nm2; tid += warpSize)
        {
            s_wsp0[warpId * nq0 * nq1 * nq2 + tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncwarp();


        //step-2 : direction 0
        for(unsigned int tid = laneId; tid < nq0 * nm1 * nm2; tid += warpSize)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();


        //step-3 : direction 1
        for(unsigned int tid = laneId; tid < nq0 * nq1 * nm2; tid += warpSize)
        {
            int q = tid / (nq0 * nm2);
            int p = (tid % (nq0 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncwarp();


        //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = laneId; tid < nq0 * nq1 * nq2; tid += warpSize)
        {
            int p = tid / (nq1 * nq2);
            int q = (tid % (nq1 * nq2)) / nq2;
            int r = tid % nq2;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
        }
        __syncwarp();


        //Reverse Operations

        //step-6 : direction 2
        for(unsigned int tid = laneId; tid < nq0 * nq1 * nm2; tid += warpSize)
        {
            int q = tid / (nq0 * nm2);
            int p = (tid % (nq0 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-7 : direction 1
        for(unsigned int tid = laneId; tid < nm1 * nm2 * nq0; tid += warpSize)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-8 : direction 0
        for(unsigned int tid = laneId; tid < nm0 * nm1 * nm2; tid += warpSize)
        {
            int i = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-9 : Copy wsp0 to out
        for(unsigned int tid = laneId; tid < nm0 * nm1 * nm2; tid += warpSize)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[warpId * nq0 * nq1 * nq2 + tid];
        }
        __syncwarp();

        e += gridDim.x * warpsPerBlock;
    }
}


template<typename T, unsigned int nq0, unsigned int nq1, unsigned int nq2>
__global__ void BwdTransHexKernel_QP_1D_Warp_Q1(
    unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_JxW, const T *__restrict__ d_in,
    T *__restrict__ d_out)
{
    unsigned int warpsPerBlock = blockDim.x / warpSize;
    unsigned int warpId        = threadIdx.x / warpSize;
    int laneId                 = threadIdx.x % warpSize;       // position of thread in warp

    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + warpsPerBlock * nq0 * nq1 * nq2;


    //copy to shared memory
    if(laneId < nq0 * nm0)
    {
        s_basis0[laneId] = d_basis0[laneId];
    }
 

    if(laneId < nq1 * nm1)
    {
        s_basis1[laneId] = d_basis1[laneId];
    }


    if(laneId < nq2 * nm2)
    {
        s_basis2[laneId] = d_basis2[laneId];
    }

    
    unsigned int e = blockIdx.x * warpsPerBlock + warpId;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        if(laneId < nm0 * nm1 * nm2)
        {
            s_wsp0[warpId * nq0 * nq1 * nq2 + laneId] = d_in[nm0 * nm1 * nm2 * e + laneId];
        }
        __syncwarp();


        //step-2 : direction 0
        if(laneId < nq0 * nm1 * nm2)
        {
            int p = laneId / (nm1 * nm2);
            int j = (laneId % (nm1 * nm2)) / nm2;
            int k = laneId % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();


        //step-3 : direction 1
        if(laneId < nq0 * nq1 * nm2)
        {
            int q = laneId / (nq0 * nm2);
            int p = (laneId % (nq0 * nm2)) / nm2;
            int k = laneId % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncwarp();


        //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
        if(laneId < nq0 * nq1 * nq2)
        {
            int p = laneId / (nq1 * nq2);
            int q = (laneId % (nq1 * nq2)) / nq2;
            int r = laneId % nq2;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
        }
        __syncwarp();

        //Reverse Operations

        //step-6 : direction 2
        if(laneId < nq0 * nq1 * nm2)
        {
            int q = laneId / (nq0 * nm2);
            int p = (laneId % (nq0 * nm2)) / nm2;
            int k = laneId % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-7 : direction 1
        if(laneId < nm1 * nm2 * nq0)
        {
            int p = laneId / (nm1 * nm2);
            int j = (laneId % (nm1 * nm2)) / nm2;
            int k = laneId % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[warpId * nq0 * nq1 * nq2 + q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-8 : direction 0
        if(laneId < nm0 * nm1 * nm2)
        {
            int i = laneId / (nm1 * nm2);
            int j = (laneId % (nm1 * nm2)) / nm2;
            int k = laneId % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[warpId * nq0 * nq1 * nq2 + p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[warpId * nq0 * nq1 * nq2 + i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-9 : Copy wsp0 to out
        if(laneId < nm0 * nm1 * nm2)
        {
            d_out[e * nm0 * nm1 * nm2 + laneId] = s_wsp0[warpId * nq0 * nq1 * nq2 + laneId];
        }
        __syncwarp();

        e += gridDim.x * warpsPerBlock;
    }
}

template<typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
__global__ void BwdTransHexKernel_QP_1D(
    const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;


    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncthreads();


        //step-2 : direction 0
        for(unsigned int tid = threadIdx.x; tid < nq0 * nm1 * nm2; tid += blockDim.x)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();


        //step-3 : direction 1
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
        {
            int q = tid / (nq0 * nm2);
            int p = (tid % (nq0 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncthreads();


        //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x)
        {
            int p = tid / (nq1 * nq2);
            int q = (tid % (nq1 * nq2)) / nq2;
            int r = tid % nq2;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
        }
        __syncthreads();


        //Reverse Operations

        //step-6 : direction 2
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
        {
            int p = tid / (nq1 * nm2);
            int q = (tid % (nq1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[p * nq1 * nm2 + q * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-7 : direction 1
        for(unsigned int tid = threadIdx.x; tid < nm1 * nm2 * nq0; tid += blockDim.x)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[p * nq1 * nm2 + q * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-8 : direction 0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            int i = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-9 : Copy wsp0 to out
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        __syncthreads();

        e += gridDim.x;
    }   
}

template<typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
__global__ void BwdTransHexKernel_QP_1D_SimpleMap(
    const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;


    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    const int tid = threadIdx.x;

    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        if(tid < nm0 * nm1 * nm2)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncthreads();


        //step-2 : direction 0
        if(tid < nq0 * nm1 * nm2)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();


        //step-3 : direction 1
        if(tid < nq0 * nq1 * nm2)
        {
            int q = tid / (nq0 * nm2);
            int p = (tid % (nq0 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncthreads();


        //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
        int p = tid / (nq1 * nq2);
        int q = (tid % (nq1 * nq2)) / nq2;
        int r = tid % nq2;

        T tmp = 0.0;
        for(unsigned int k = 0; k < nm2; ++k)
        {
            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
        }
        s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];

        __syncthreads();


        //Reverse Operations

        //step-6 : direction 2
        if(tid < nq0 * nq1 * nm2)
        {
            int p = tid / (nq1 * nm2);
            int q = (tid % (nq1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[p * nq1 * nm2 + q * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-7 : direction 1
        if(tid < nm1 * nm2 * nq0)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[p * nq1 * nm2 + q * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-8 : direction 0
        if(tid < nm0 * nm1 * nm2)
        {
            int i = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-9 : Copy wsp0 to out
        for(unsigned int tidx = threadIdx.x; tidx < nm0 * nm1 * nm2; tidx += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tidx] = s_wsp0[tidx];
        }
        __syncthreads();

        e += gridDim.x;
    }   
}

    // In 3D thread-blocks in CUDA, X dimension is fastest, Z is slowest
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
    // threadIdx.x for i and p
    // threadIdx.x for j and q
    // threadIdx.x for k and r

    template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS(
        const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        const unsigned int nm0 = nq0 - 1;
        const unsigned int nm1 = nq1 - 1;
        const unsigned int nm2 = nq2 - 1;

        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    
        //Finding global indices
        unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
        unsigned int linearThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

        //copy to shared memory
        for(unsigned int tid = linearThreadIdx; tid < nq0 * nm0; tid += blockSize)
        {
            s_basis0[tid] = d_basis0[tid];
        }
    
        for(unsigned int tid = linearThreadIdx; tid < nq1 * nm1; tid += blockSize)
        {
            s_basis1[tid] = d_basis1[tid];
        }
    
        for(unsigned int tid = linearThreadIdx; tid < nq2 * nm2; tid += blockSize)
        {
            s_basis2[tid] = d_basis2[tid];
        }
    
        
        unsigned int e = blockIdx.x;
    
        while(e < nelmt)
        {
            //step-1 : Copy from in to the wsp0
            for(unsigned int tid = linearThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            __syncthreads();
    
    
            //step-2 : direction 0
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int i = 0; i < nm0; ++i)
                        {
                            tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                        }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
                        
                    }
                }
            }
            __syncthreads();
    
    
            //step-3 : direction 1
            for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int j = 0; j < nm1; j++)
                        {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                        }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    

            //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                    for(unsigned int r = threadIdx.z; r < nq2; r += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int k = 0; k < nm2; ++k)
                        {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                        }
                        s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
                    }
                }
            }
            __syncthreads();       

            //Reverse Operations
      
            //step-6 : direction 2
            for(unsigned int q = threadIdx.y; q < nq1; q += blockDim.y){
                for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){
                    
                        T tmp = 0.0;
                        for(unsigned int r = 0; r < nq2; ++r)
                        {
                            tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                        }
                        s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    
            //step-7 : direction 1
            for(unsigned int p = threadIdx.x; p < nq0; p += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){    

                        T tmp = 0.0;
                        for(unsigned int q = 0; q < nq1; q++)
                        {
                            tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                        }
                        s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                    }
                }
            }
            __syncthreads();
    
            //step-8 : direction 0
            for(unsigned int i = threadIdx.x; i < nm0; i += blockDim.x){
                for(unsigned int j = threadIdx.y; j < nm1; j += blockDim.y){
                    for(unsigned int k = threadIdx.z; k < nm2; k += blockDim.z){

                        T tmp = 0.0;
                        for(unsigned int p = 0; p < nq0; ++p)
                        {
                            tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                        }
                        s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;       
                    }
                }
            }
            __syncthreads();


            //step-9 : Copy wsp0 to out
            for(unsigned int tid = linearThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
            {
                d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
            } 
            __syncthreads();
    
            e += gridDim.x;
        }
    }


    template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
    __global__ void BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap(
        const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, T *__restrict__ d_in, T *__restrict__ d_out)
    {
        const unsigned int nm0 = nq0 - 1;
        const unsigned int nm1 = nq1 - 1;
        const unsigned int nm2 = nq2 - 1;
        
        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;
    
        //Finding global indices
        int linearThreadIdx = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
        int blockSize = blockDim.x * blockDim.y * blockDim.z;
        
        //copy to shared memory
        for(unsigned int tid = linearThreadIdx; tid < nm0 * nq0; tid += blockSize)
        {
            s_basis0[tid] = d_basis0[tid];
        }

        for(unsigned int tid = linearThreadIdx; tid < nm1 * nq1; tid += blockSize)
        {
            s_basis1[tid] = d_basis1[tid];
        }

        for(unsigned int tid = linearThreadIdx; tid < nm2 * nq2; tid += blockSize)
        {
            s_basis2[tid] = d_basis2[tid];
        }
        

        unsigned int e = blockIdx.x;
        
        while(e < nelmt)
        {   
            //step-1 : Copy from in to the wsp0
            if(linearThreadIdx < nm0 * nm1 * nm2)
            {
                s_wsp0[linearThreadIdx] = d_in[e * nm0 * nm1 * nm2 + linearThreadIdx];
            }
            __syncthreads();
    
            //step-2 : direction 0
            if(threadIdx.y < nm1 && threadIdx.z < nm2){
                int p = threadIdx.x;
                int j = threadIdx.y;
                int k = threadIdx.z;

                T tmp = 0.0;
                for(unsigned int i = 0; i < nm0; ++i)
                {
                    tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                }
                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
            }
            __syncthreads();
    
            //step-3 : direction 1
            if(threadIdx.z < nm2){ 

                int q = threadIdx.y;
                int p = threadIdx.x;
                int k = threadIdx.z;
                
                T tmp = 0.0;
                for(unsigned int j = 0; j < nm1; j++)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                }
                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            __syncthreads();
    

            //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
            {
            int p = threadIdx.x;
            int q = threadIdx.y;
            int r = threadIdx.z;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp * d_JxW[e * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            }
            __syncthreads();
            
      
            //step-6 : direction 2
            if(threadIdx.z < nm2)
            {
                int p = threadIdx.x;
                int q = threadIdx.y;
                int k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int r = 0; r < nq2; ++r)
                {
                    tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                }
                s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-7 : direction 1
            if(threadIdx.y < nm1  && threadIdx.z < nm2)
            {
                int p = threadIdx.x;
                int j = threadIdx.y;
                int k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int q = 0; q < nq1; q++)
                {
                    tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                }
                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-8 : direction 0
            if(threadIdx.x < nm0 && threadIdx.y < nm1 && threadIdx.z < nm2)
            {
                int i = threadIdx.x;
                int j = threadIdx.y;
                int k = threadIdx.z;
            
                T tmp = 0.0;
                for(unsigned int p = 0; p < nq0; ++p)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                }
                s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
            }
            __syncthreads();

            //step-9 : Copy wsp0 to out
            if(linearThreadIdx < nm0 * nm1 * nm2)
            {
                d_out[e * nm0 * nm1 * nm2 + linearThreadIdx] = s_wsp0[linearThreadIdx];
            } 
            __syncthreads();
    
            e += gridDim.x;
        }
    }


    template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
    __global__ void BwdTransHexKernel_QP_1D_2D_BLOCKS_pq(
        const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        const unsigned int nm0 = nq0 - 1;
        const unsigned int nm1 = nq1 - 1;
        const unsigned int nm2 = nq2 - 1;

        T r_p[nq0];
        T r_q[nq1];
        T r_r[nq2];

        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

        //copy to shared memory
        for(unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x )
        {
            s_basis0[tid] = d_basis0[tid];
        }

        for(unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x )
        {
            s_basis1[tid] = d_basis1[tid];
        }

        for(unsigned int tid = threadIdx.x; tid < nm2 * nq2; tid += blockDim.x )
        {
            s_basis2[tid] = d_basis2[tid];
        }


        int e = blockIdx.x;

        while(e < nelmt)
        {   
            //register for dot product ops
            T r_tmp = 0; 
            
            //step-1 : Copy from in to the wsp0
            for(int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
            {
                s_wsp0[tid] = d_in[e * nm0 * nm1 * nm2 + tid];
            }
            __syncthreads();

            //step-2 : direction 0
            for(int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x){

                const int p = tid / nm1;
                const int j = tid % nm1;
            
                //copy to register
                for(int i = 0; i < nm0; ++i){
                    r_p[i] = s_basis0[p * nm0 + i];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int i = 0; i < nm0; ++i){
                       r_tmp += r_p[i] * s_wsp0[k * nm0 * nm1 + i * nm1 + j];
                    }
                    s_wsp1[k * nm1 * nq0 + p * nm1 + j] = r_tmp;
                }
            }
            __syncthreads();


            //step-3 : direction 1
            for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x){
            
                const int p = tid / nq1;
                const int q = tid % nq1;

                //copy to register
                for(int j = 0; j < nm1; ++j){
                    r_q[j] = s_basis1[q * nm1 + j];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int j = 0; j < nm1; ++j){
                        r_tmp += r_q[j] * s_wsp1[k * nm1 * nq0 + p * nm1 + j];
                    }
                    s_wsp0[k * nq0 * nq1 + p * nq1 + q] = r_tmp;
                }
            }
            __syncthreads();


            //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
            for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
            {
                const int p = tid / nq1;
                const int q = tid % nq1;

                //copy to register
                for(int k = 0; k < nm2; ++k){
                    r_r[k] = s_wsp0[k * nq0 * nq1 + p * nq1 + q];
                }

                //mat-vec multp
                for(int r = 0; r < nq2; ++r){
                    r_tmp = 0;
                    for(int k = 0; k < nm2; ++k){
                        r_tmp += r_r[k] * s_basis2[r * nm2 + k];
                    }
                    s_wsp1[r * nq0 * nq1 + p * nq1 + q] = r_tmp * d_JxW[e * nq0 * nq1 * nq2 + r * nq0 * nq1 + p * nq1 + q];
                }
            }
            __syncthreads();

            //Reverse Operations
    
            //step-6 : direction 2
            for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
            {
                const int p = tid / nq1;
                const int q = tid % nq1;

                //copy to register
                for(int r = 0; r < nq2; ++r){
                    r_r[r] = s_wsp1[r * nq0 * nq1 + p * nq1 + q];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int r = 0; r < nq2; ++r){
                        r_tmp += r_r[r] * s_basis2[r * nm2 + k];
                    }
                    s_wsp0[k * nq0 * nq1 + p * nq1 + q] = r_tmp;
                }
            }
            __syncthreads();


            //step-7 : direction 1
            for(int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x)
            {
                const int p = tid / nm1;
                const int j = tid % nm1;

                //copy to register
                for(int q = 0; q < nq1; ++q){
                    r_q[q] = s_basis1[q * nm1 + j];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int q = 0; q < nq1; ++q){
                        r_tmp += r_q[q] * s_wsp0[k * nq0 * nq1 + p * nq1 + q];
                    }
                    s_wsp1[k * nq0 * nm1 + p * nm1 + j] = r_tmp;
                }
            }
            __syncthreads();


            //step-8 : direction 0
            for(int tid = threadIdx.x; tid < nm0 * nm1; tid += blockDim.x)
            {
                const int i = tid / nm1;
                const int j = tid % nm1;

                //copy to register
                for(int p = 0; p < nq0; ++p){
                    r_p[p] = s_basis0[p * nm0 + i];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int p = 0; p < nq0; ++p){
                        r_tmp += r_p[p] *  s_wsp1[k * nq0 * nm1 + p * nm1 + j];
                    }
                    s_wsp0[k * nm1 * nm0 + j * nm0 + i] = r_tmp;
                }
            }
            __syncthreads();


            //step-9 : Copy wsp0 to out
            for(int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
            {
                d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
            } 
            __syncthreads();
        
            e += gridDim.x;
        }
    }   


     template <typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
    __global__ void BwdTransHexKernel_QP_1D_2D_BLOCKS_pq_SimpleMap(
        const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
        const T *__restrict__ d_basis2, const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        const unsigned int nm0 = nq0 - 1;
        const unsigned int nm1 = nq1 - 1;
        const unsigned int nm2 = nq2 - 1;

        T r_p[nq0];
        T r_q[nq1];
        T r_r[nq2];

        extern __shared__ T shared[];
        T *s_basis0 = shared;
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

        //copy to shared memory
        for(unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x )
        {
            s_basis0[tid] = d_basis0[tid];
        }

        for(unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x )
        {
            s_basis1[tid] = d_basis1[tid];
        }

        for(unsigned int tid = threadIdx.x; tid < nm2 * nq2; tid += blockDim.x )
        {
            s_basis2[tid] = d_basis2[tid];
        }


        int e = blockIdx.x;

        while(e < nelmt)
        {   
            const int tid = threadIdx.x;

            //register for dot product ops
            T r_tmp = 0; 
            
            //step-1 : Copy from in to the wsp0
            for(int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
            {
                s_wsp0[tid] = d_in[e * nm0 * nm1 * nm2 + tid];
            }
            __syncthreads();

            //step-2 : direction 0
            if(tid < nq0 * nm1)
            {
                const int p = tid / nm1;
                const int j = tid % nm1;
            
                //copy to register
                for(int i = 0; i < nm0; ++i){
                    r_p[i] = s_basis0[p * nm0 + i];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int i = 0; i < nm0; ++i){
                       r_tmp += r_p[i] * s_wsp0[k * nm0 * nm1 + i * nm1 + j];
                    }
                    s_wsp1[k * nm1 * nq0 + p * nm1 + j] = r_tmp;
                }
            }
            __syncthreads();


            //step-3 : direction 1
            const int q = tid / nq0;
            const int p = tid % nq0;

            //copy to register
            for(int j = 0; j < nm1; ++j){
                r_q[j] = s_basis1[q * nm1 + j];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int j = 0; j < nm1; ++j){
                    r_tmp += r_q[j] * s_wsp1[k * nm1 * nq0 + p * nm1 + j];
                }
                s_wsp0[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
            }
            __syncthreads();


            //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi

            //copy to register
            for(int k = 0; k < nm2; ++k){
                r_r[k] = s_wsp0[k * nq0 * nq1 + q * nq0 + p];
            }

            //mat-vec multp
            for(int r = 0; r < nq2; ++r){
                r_tmp = 0;
                for(int k = 0; k < nm2; ++k){
                    r_tmp += r_r[k] * s_basis2[r * nm2 + k];
                }
                s_wsp1[r * nq0 * nq1 + q * nq0 + p] = r_tmp * d_JxW[e * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            }
            __syncthreads();

            //Reverse Operations
    
            //step-6 : direction 2

            //copy to register
            for(int r = 0; r < nq2; ++r){
                r_r[r] = s_wsp1[r * nq0 * nq1 + q * nq0 + p];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k)
            {
                r_tmp = 0;
                for(int r = 0; r < nq2; ++r){
                    r_tmp += r_r[r] * s_basis2[r * nm2 + k];
                }
                s_wsp0[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
            }
            __syncthreads();


            //step-7 : direction 1
            if(tid < nq0 * nm1)
            {
                int p = tid / nm1;
                int j = tid % nm1;

                //copy to register
                for(int q = 0; q < nq1; ++q){
                    r_q[q] = s_basis1[q * nm1 + j];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int q = 0; q < nq1; ++q){
                        r_tmp += r_q[q] * s_wsp0[k * nq0 * nq1 + q * nq0 + p];
                    }
                    s_wsp1[k * nq0 * nm1 + p * nm1 + j] = r_tmp;
                }
            }
            __syncthreads();

            //step-8 : direction 0
            if(tid < nm0 * nm1)
            {
                const int i = tid / nm1;
                const int j = tid % nm1;

                //copy to register
                for(int p = 0; p < nq0; ++p){
                    r_p[p] = s_basis0[p * nm0 + i];
                }

                //mat-vec multp
                for(int k = 0; k < nm2; ++k){
                    r_tmp = 0;
                    for(int p = 0; p < nq0; ++p){
                        r_tmp += r_p[p] * s_wsp1[k * nq0 * nm1 + p * nm1 + j];
                    }
                    s_wsp0[k * nm0 * nm1 + i * nm1 + j] = r_tmp;
                }
            }
            __syncthreads();


            //step-9 : Copy wsp0 to out
            for(int tidx = threadIdx.x; tidx < nm0 * nm1 * nm2; tidx += blockDim.x)
            {
                d_out[e * nm0 * nm1 * nm2 + tidx] = s_wsp0[tidx];
            }
            __syncthreads();

            e += gridDim.x;
        }
    }   

} //namespace Parallel
} //namespace BK1

#endif //BK1_TEMPLATED_CUDA_KERNELS_CUH