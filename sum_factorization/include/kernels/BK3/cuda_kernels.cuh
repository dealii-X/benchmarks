#ifndef BK3_CUDA_KERNELS_CUH
#define BK3_CUDA_KERNELS_CUH

#include <timer.hpp>
#include <vector>

namespace BK3{
namespace Parallel{

template<typename T>
__global__ void TransHexKernel_QP_3D_Block(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ d_basis0, const T *__restrict__ d_basis1, const T *__restrict__ d_basis2, 
    const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    extern __shared__ T shared[];
    T *s_basis0  = shared;
    T *s_basis1  = s_basis0 + nm0 * nq0;
    T *s_basis2  = s_basis1 + nm1 * nq1;
    T *s_dbasis0 = s_basis2 + nm2 * nq2;
    T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
    T *s_dbasis2 = s_dbasis1 + nq1 * nq1;
    T *rqr       = s_dbasis2 + nq2 * nq2;
    T *rqs       = rqr + nq0 * nq1 * nq2;
    T *rqt       = rqs + nq0 * nq1 * nq2;
    T *s_wsp0    = rqt + nq0 * nq1 * nq2;
    T *s_wsp1    = s_wsp0 + nq0 * nq1 * nq2;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nm0 * nq0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nm1 * nq1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nm2 * nq2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }


    for(unsigned int tid = threadIdx.x; tid < nq0 * nq0; tid += blockDim.x)
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nq1; tid += blockDim.x)
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nq2; tid += blockDim.x)
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }

    __syncthreads();

    /*
    Interpolate to GL nodes
    */

    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            s_wsp0[tid] = d_in[e * nm0 * nm1 * nm2 + tid];
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

        //step-4 : direction 2
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
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        __syncthreads();

        //Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;
        T qr, qs, qt;

        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x){

            unsigned int p = tid / (nq1 * nq2);
            unsigned int q = (tid % (nq1 * nq2)) / nq2;
            unsigned int r = tid % nq2;

            qr = 0; qs = 0; qt = 0;
    
            //step-5 : Load Geometric Factors, coalesced access
            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + p * nq1 * nq2 + q * nq2 + r];
    
            //step-6 : Multiply by D
            for(unsigned int n = 0; n < nq0; n++){
                qr += s_wsp1[n * nq1 * nq2 + q * nq2 + r] * s_dbasis0[p * nq0 + n];
            }
    
            for(unsigned int n = 0; n < nq1; n++){
                qs += s_wsp1[p * nq1 * nq2 + n * nq2 + r] * s_dbasis1[q * nq1 + n];
            }
            
            for(unsigned int n = 0; n < nq2; n++){
                qt += s_wsp1[p * nq1 * nq2 + q * nq2 + n] * s_dbasis2[r * nq2 + n];
            }
            
            // step-7 : Apply chain rule
            rqr[p * nq1 * nq2 + q * nq2 + r] = Grr * qr + Grs * qs + Grt * qt;
            rqs[p * nq1 * nq2 + q * nq2 + r] = Grs * qr + Gss * qs + Gst * qt;
            rqt[p * nq1 * nq2 + q * nq2 + r] = Grt * qr + Gst * qs + Gtt * qt;
        }
        __syncthreads();

        // step-8 : Compute out vector in GL nodes
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nq2; tid += blockDim.x){

            int p = tid / (nq1 * nq2);
            int q = (tid % (nq1 * nq2)) / nq2;
            int r = tid % nq2;

            T tmp0 = 0;
            for(unsigned int n = 0; n < nq0; ++n)
                tmp0 += rqr[n * nq1 * nq2 + q * nq2 + r] * s_dbasis0[n * nq0 + p];
    
            for(unsigned int n = 0; n < nq1; ++n)                
                tmp0 += rqs[p * nq1 * nq2 + n * nq2 + r] * s_dbasis1[n * nq1 + q];
    
            for(unsigned int n = 0; n < nq2; ++n)
                tmp0 += rqt[p * nq1 * nq2 + q * nq2 + n] * s_dbasis2[n * nq2 + r];
    
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp0;
        }

        __syncthreads();

        /*
        Interpolate to GLL nodes
        */
        
        //step-9 : direction 2
        for(unsigned int tid = threadIdx.x; tid < nq0 * nq1 * nm2; tid += blockDim.x)
        {
            int q = tid / (nq0 * nm2);
            int p = (tid % (nq0 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-10 : direction 1
        for(unsigned int tid = threadIdx.x; tid < nm1 * nm2 * nq0; tid += blockDim.x)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncthreads();

        //step-11 : direction 0
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

        //step-12 : Copy wsp0 to out
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        __syncthreads();

        e += gridDim.x;
    }
}


template<typename T>
__global__ void TransHexKernel_QP_3D_Block_SimpleMap(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ d_basis0, const T *__restrict__ d_basis1, const T *__restrict__ d_basis2, 
    const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    extern __shared__ T shared[];
    T *s_basis0  = shared;
    T *s_basis1  = s_basis0 + nm0 * nq0;
    T *s_basis2  = s_basis1 + nm1 * nq1;
    T *s_dbasis0 = s_basis2 + nm2 * nq2;
    T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
    T *s_dbasis2 = s_dbasis1 + nq1 * nq1;
    T *rqr       = s_dbasis2 + nq2 * nq2;
    T *rqs       = rqr + nq0 * nq1 * nq2;
    T *rqt       = rqs + nq0 * nq1 * nq2;
    T *s_wsp0    = rqt + nq0 * nq1 * nq2;
    T *s_wsp1    = s_wsp0 + nq0 * nq1 * nq2;


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


    for(unsigned int tid = linearThreadIdx; tid < nq0 * nq0; tid += blockSize)
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(unsigned int tid = linearThreadIdx; tid < nq1 * nq1; tid += blockSize)
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(unsigned int tid = linearThreadIdx; tid < nq2 * nq2; tid += blockSize)
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }

    __syncthreads();

    /*
    Interpolate to GL nodes
    */

    int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        if(linearThreadIdx < nm0 * nm1 * nm2)
        {
            s_wsp0[linearThreadIdx] = d_in[e * nm0 * nm1 * nm2 + linearThreadIdx];
        }
        __syncthreads();

        //step-2 : direction 0
        if(threadIdx.z < nm2)
        {
            int p = threadIdx.x;
            int j = threadIdx.y;
            int k = threadIdx.z;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[k * nm0 * nm1 + j * nm0 + i] * s_basis0[p * nm0 + i];
            }
            s_wsp1[k * nm1 * nq0 + j * nq0 + p] = tmp;
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
                tmp += s_wsp1[k * nm1 * nq0 + j * nq0 + p] * s_basis1[q * nm1 + j];
            }
            s_wsp0[k * nq0 * nq1 + p * nq1 + q] = tmp;
        }
        __syncthreads();
    
        //step-4 : direction 2

        int p = threadIdx.x;
        int q = threadIdx.y;
        int r = threadIdx.z;
            
        T tmp = 0.0;
        for(unsigned int k = 0; k < nm2; ++k)
        {
            tmp += s_wsp0[k * nq0 * nq1 + p * nq1 + q] * s_basis2[r * nm2 + k];
        }
        s_wsp1[r * nq0 * nq1 + q * nq0 + p] = tmp;
        
        __syncthreads();

        //Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;
        T qr, qs, qt;

        qr = 0; qs = 0; qt = 0;

        //step-5 : Load Geometric Factors, coalesced access
        Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
    
        //step-6 : Multiply by D
        for(unsigned int n = 0; n < nq0; n++){
            qr += s_wsp1[r * nq0 * nq1 + q * nq0 + n] * s_dbasis0[p * nq0 + n];
        }
    
        for(unsigned int n = 0; n < nq1; n++){
            qs += s_wsp1[r * nq0 * nq1 + n * nq0 + p] * s_dbasis1[q * nq1 + n];
        }
            
        for(unsigned int n = 0; n < nq2; n++){
            qt += s_wsp1[n * nq0 * nq1 + q * nq0 + p] * s_dbasis2[r * nq2 + n];
        }
            
        // step-7 : Apply chain rule
        rqr[p * nq1 * nq2 + q * nq2 + r] = Grr * qr + Grs * qs + Grt * qt;
        rqs[p * nq1 * nq2 + q * nq2 + r] = Grs * qr + Gss * qs + Gst * qt;
        rqt[p * nq1 * nq2 + q * nq2 + r] = Grt * qr + Gst * qs + Gtt * qt;

        __syncthreads();

        // step-8 : Compute out vector in GL nodes
        
        T tmp0 = 0;
        for(unsigned int n = 0; n < nq0; ++n)
            tmp0 += rqr[n * nq1 * nq2 + q * nq2 + r] * s_dbasis0[n * nq0 + p];
    
        for(unsigned int n = 0; n < nq1; ++n)                
            tmp0 += rqs[p * nq1 * nq2 + n * nq2 + r] * s_dbasis1[n * nq1 + q];
    
        for(unsigned int n = 0; n < nq2; ++n)
            tmp0 += rqt[p * nq1 * nq2 + q * nq2 + n] * s_dbasis2[n * nq2 + r];
    
        s_wsp1[r * nq0 * nq1 + q * nq0 + p] = tmp0;
        
        __syncthreads();


        /*
        Interpolate to GLL nodes
        */
        
        //step-9 : direction 2
        if(threadIdx.z < nm2)
        {
            int p = threadIdx.x;
            int q = threadIdx.y;
            int k = threadIdx.z;
            
            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[r * nq0 * nq1 + q * nq0 + p] * s_basis2[r * nm2 + k];
            }
            s_wsp0[k * nq0 * nq1 + q * nq0 + p] = tmp;
        }
        __syncthreads();

        //step-10 : direction 1
        if(threadIdx.y < nm1)
        {
            int p = threadIdx.x;
            int j = threadIdx.y;
            int k = threadIdx.z;
            
            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[k * nq0 * nq1 + q * nq0 + p]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[k * nm1 * nq0 + j * nq0 + p] = tmp;
        }
        __syncthreads();

        //step-11 : direction 0
        if(threadIdx.x < nm0)
        {
            int i = threadIdx.x;
            int j = threadIdx.y;
            int k = threadIdx.z;
            
            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[k * nm1 * nq0 + j * nq0 + p] * s_basis0[p * nm0 + i];
            }
            s_wsp0[k * nm1 * nm0 + j * nm0 + i] = tmp;
        }
        __syncthreads();

        //step-12 : Copy wsp0 to out
        if(linearThreadIdx < nm0 * nm1 * nm2)
        {
            d_out[e * nm0 * nm1 * nm2 + linearThreadIdx] = s_wsp0[linearThreadIdx];
        } 
        __syncthreads();
    
        e += gridDim.x;
    }
}


template<typename T>
__global__ void TransHexKernel_QP_2D_Block_pq(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ d_basis0, const T *__restrict__ d_basis1, const T *__restrict__ d_basis2, 
    const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    T r_p[10];
    T r_q[10];
    T r_r[10];

    extern __shared__ T shared[];
    T *s_basis0  = shared;
    T *s_basis1  = shared   + nq0 * nm0;
    T *s_basis2  = s_basis1 + nq1 * nm1;

    T *s_dbasis0 = s_basis2  + nq2 * nm2;
    T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
    T *s_dbasis2 = s_dbasis1 + nq1 * nq1;

    T *rqr       = s_dbasis2 + nq2 * nq2;
    T *rqs       = rqr       + nq0 * nq1 * nq2;
    T *rqt       = rqs       + nq0 * nq1 * nq2;

    T *s_wsp0    = rqt    + nq0 * nq1 * nq2;
    T *s_wsp1    = s_wsp0 + nq0 * nq1 * nq2;


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


    for(unsigned int tid = threadIdx.x; tid < nq0 * nq0; tid += blockDim.x)
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nq1; tid += blockDim.x)
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nq2; tid += blockDim.x)
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }
    __syncthreads();

    /*
    Interpolate to GL nodes
    */

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

            int p = tid / nm1;
            int j = tid % nm1;
    
            //copy to register
            for(int i = 0; i < nm0; ++i){
                r_p[i] = s_basis0[p * nm0 + i];
            }
            
            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int i = 0; i < nm0; ++i){
                   r_tmp += r_p[i] * s_wsp0[k * nm0 * nm1 + j * nm0 + i];
                }
                s_wsp1[k * nm1 * nq0 + j * nq0 + p] = r_tmp;
            }
        }
        __syncthreads();


        //step-3 : direction 1
        for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x){
        
            int p = tid / nq1;
            int q = tid % nq1;

            //copy to register
            for(int j = 0; j < nm1; ++j){
                r_q[j] = s_basis1[q * nm1 + j];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int j = 0; j < nm1; ++j){
                    r_tmp += r_q[j] * s_wsp1[k * nm1 * nq0 + j * nq0 + p];
                }
                s_wsp0[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
            }
        }
        __syncthreads();


        //step-4 : direction 2
        for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
        {
            int p = tid / nq1;
            int q = tid % nq1;

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
                s_wsp1[r * nq0 * nq1 + q * nq0 + p] = r_tmp;
            }
        }
        __syncthreads();

        for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x){

            int q = tid / nq0;
            int p = tid % nq0;

            //Geometric vals
            T Grr, Grs, Grt, Gss, Gst, Gtt;
            T qr, qs, qt;
    
            for(int r = 0; r < nq2; ++r){
                qr = 0; qs = 0; qt = 0;
                
                //step-5 : Load Geometric Factors, coalesced access
                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            
                //step-6 : Multiply by D
                for(unsigned int n = 0; n < nq0; n++){
                    qr += s_wsp1[r * nq0 * nq1 + q * nq0 + n] * s_dbasis0[p * nq0 + n];
                }
            
                for(unsigned int n = 0; n < nq1; n++){
                    qs += s_wsp1[r * nq0 * nq1 + n * nq0 + p] * s_dbasis1[q * nq1 + n];
                }
                    
                for(unsigned int n = 0; n < nq2; n++){
                    qt += s_wsp1[n * nq0 * nq1 + q * nq0 + p] * s_dbasis2[r * nq2 + n];
                }
                    
                // step-7 : Apply chain rule
                rqr[p * nq1 * nq2 + q * nq2 + r] = Grr * qr + Grs * qs + Grt * qt;
                rqs[p * nq1 * nq2 + q * nq2 + r] = Grs * qr + Gss * qs + Gst * qt;
                rqt[p * nq1 * nq2 + q * nq2 + r] = Grt * qr + Gst * qs + Gtt * qt;
            }
        }
        __syncthreads();
    
    
        // step-8 : Compute out vector in GL nodes
        for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x){

            int p = tid / nq1;
            int q = tid % nq1;

            for(int r = 0; r < nq2; ++r){
                
                T tmp0 = 0;

                for(unsigned int n = 0; n < nq0; ++n)
                    tmp0 += rqr[n * nq1 * nq2 + q * nq2 + r] * s_dbasis0[n * nq0 + p];
                
                for(unsigned int n = 0; n < nq1; ++n)                
                    tmp0 += rqs[p * nq1 * nq2 + n * nq2 + r] * s_dbasis1[n * nq1 + q];
                
                for(unsigned int n = 0; n < nq2; ++n)
                    tmp0 += rqt[p * nq1 * nq2 + q * nq2 + n] * s_dbasis2[n * nq2 + r];
                    
                s_wsp0[r * nq0 * nq1 + q * nq0 + p] = tmp0;
            }
        }
        __syncthreads();


        /*
        Interpolate to GLL nodes
        */
        
        //step-9 : direction 2
        for(int tid = threadIdx.x; tid < nq0 * nq1; tid += blockDim.x)
        {
            int p = tid / nq1;
            int q = tid % nq1;

            //copy to register
            for(int r = 0; r < nq2; ++r){
                r_r[r] = s_wsp0[r * nq0 * nq1 + q * nq0 + p];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int r = 0; r < nq2; ++r){
                    r_tmp += r_r[r] * s_basis2[r * nm2 + k];
                }
                s_wsp1[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
            }
        }
        __syncthreads();


        //step-10 : direction 1
        for(int tid = threadIdx.x; tid < nq0 * nm1; tid += blockDim.x)
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
                    r_tmp += r_q[q] * s_wsp1[k * nq0 * nq1 + q * nq0 + p];
                }
                s_wsp0[p * nm1 * nm2 + j * nm2 + k] = r_tmp;
            }
        }
        __syncthreads();


        //step-11 : direction 0
        for(int tid = threadIdx.x; tid < nm0 * nm1; tid += blockDim.x)
        {
            int i = tid / nm1;
            int j = tid % nm1;

            //copy to register
            for(int p = 0; p < nq0; ++p){
                r_p[p] = s_basis0[p * nm0 + i];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int p = 0; p < nq0; ++p){
                    r_tmp += r_p[p] * s_wsp0[p * nm1 * nm2 + j * nm2 + k];
                }
                s_wsp1[i * nm1 * nm2 + j * nm2 + k] = r_tmp;
            }
        }
        __syncthreads();


        //step-12 : Copy wsp0 to out
        for(int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp1[tid];
        } 
        __syncthreads();
    
        e += gridDim.x;
    }
}

template<typename T>
__global__ void TransHexKernel_QP_2D_Block_pq_SimpleMap(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, const unsigned int nelmt,
    const T *__restrict__ d_basis0, const T *__restrict__ d_basis1, const T *__restrict__ d_basis2, 
    const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    T r_p[10];
    T r_q[10];
    T r_r[10];

    extern __shared__ T shared[];
    T *s_basis0  = shared;
    T *s_basis1  = shared   + nq0 * nm0;
    T *s_basis2  = s_basis1 + nq1 * nm1;

    T *s_dbasis0 = s_basis2  + nq2 * nm2;
    T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
    T *s_dbasis2 = s_dbasis1 + nq1 * nq1;

    T *rqr       = s_dbasis2 + nq2 * nq2;
    T *rqs       = rqr       + nq0 * nq1 * nq2;
    T *rqt       = rqs       + nq0 * nq1 * nq2;

    T *s_wsp0    = rqt    + nq0 * nq1 * nq2;
    T *s_wsp1    = s_wsp0 + nq0 * nq1 * nq2;

    

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


    for(unsigned int tid = threadIdx.x; tid < nq0 * nq0; tid += blockDim.x)
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nq1; tid += blockDim.x)
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nq2; tid += blockDim.x)
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }
    __syncthreads();

    /*
    Interpolate to GL nodes
    */

    int e = blockIdx.x;

    while(e < nelmt)
    {   
        const int tid = threadIdx.x;

        //register for dot product ops
        T r_tmp = 0; 
        
        //step-1 : Copy from in to the wsp0
        for(int tidx = threadIdx.x; tidx < nm0 * nm1 * nm2; tidx += blockDim.x){
            s_wsp0[tidx] = d_in[e * nm0 * nm1 * nm2 + tid];
        }
        __syncthreads();

        //step-2 : direction 0
        if(tid < nq0 * nm1)
        {
            int p = tid / nm1;
            int j = tid % nm1;
    
            //copy to register
            for(int i = 0; i < nm0; ++i){
                r_p[i] = s_basis0[p * nm0 + i];
            }
            
            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int i = 0; i < nm0; ++i){
                   r_tmp += r_p[i] * s_wsp0[k * nm0 * nm1 + j * nm0 + i];
                }
                s_wsp1[k * nm1 * nq0 + j * nq0 + p] = r_tmp;
            }
        }
        __syncthreads();


        //step-3 : direction 1
        int q = tid / nq0;
        int p = tid % nq0;

        //copy to register
        for(int j = 0; j < nm1; ++j){
            r_q[j] = s_basis1[q * nm1 + j];
        }

        //mat-vec multp
        for(int k = 0; k < nm2; ++k){
            r_tmp = 0;
            for(int j = 0; j < nm1; ++j){
                r_tmp += r_q[j] * s_wsp1[k * nm1 * nq0 + j * nq0 + p];
            }
            s_wsp0[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
        }
        __syncthreads();


        //step-4 : direction 2

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
            s_wsp1[r * nq0 * nq1 + q * nq0 + p] = r_tmp;
        }
        __syncthreads();


        //Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;
        T qr, qs, qt;
    
        for(int r = 0; r < nq2; ++r){
            qr = 0; qs = 0; qt = 0;
                
            //step-5 : Load Geometric Factors, coalesced access
            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
            
            //step-6 : Multiply by D
            for(unsigned int n = 0; n < nq0; n++){
                qr += s_wsp1[r * nq0 * nq1 + q * nq0 + n] * s_dbasis0[p * nq0 + n];
            }
            
            for(unsigned int n = 0; n < nq1; n++){
                qs += s_wsp1[r * nq0 * nq1 + n * nq0 + p] * s_dbasis1[q * nq1 + n];
            }
                    
            for(unsigned int n = 0; n < nq2; n++){
                qt += s_wsp1[n * nq0 * nq1 + q * nq0 + p] * s_dbasis2[r * nq2 + n];
            }
                    
            // step-7 : Apply chain rule
            rqr[p * nq1 * nq2 + q * nq2 + r] = Grr * qr + Grs * qs + Grt * qt;
            rqs[p * nq1 * nq2 + q * nq2 + r] = Grs * qr + Gss * qs + Gst * qt;
            rqt[p * nq1 * nq2 + q * nq2 + r] = Grt * qr + Gst * qs + Gtt * qt;
        }
        __syncthreads();
    
        // step-8 : Compute out vector in GL nodes
        for(int r = 0; r < nq2; ++r){
                
            T tmp0 = 0;

            for(unsigned int n = 0; n < nq0; ++n)
                tmp0 += rqr[n * nq1 * nq2 + q * nq2 + r] * s_dbasis0[n * nq0 + p];
                
            for(unsigned int n = 0; n < nq1; ++n)                
                tmp0 += rqs[p * nq1 * nq2 + n * nq2 + r] * s_dbasis1[n * nq1 + q];
                
            for(unsigned int n = 0; n < nq2; ++n)
                tmp0 += rqt[p * nq1 * nq2 + q * nq2 + n] * s_dbasis2[n * nq2 + r];
                    
            s_wsp0[r * nq0 * nq1 + q * nq0 + p] = tmp0;

        }
        __syncthreads();

        /*
        Interpolate to GLL nodes
        */
        
        //step-9 : direction 2

        //copy to register
        for(int r = 0; r < nq2; ++r){
            r_r[r] = s_wsp0[r * nq0 * nq1 + q * nq0 + p];
        }

        //mat-vec multp
        for(int k = 0; k < nm2; ++k)
        {
            r_tmp = 0;
            for(int r = 0; r < nq2; ++r){
                r_tmp += r_r[r] * s_basis2[r * nm2 + k];
            }
            s_wsp1[k * nq0 * nq1 + q * nq0 + p] = r_tmp;
        }
        __syncthreads();

        //step-10 : direction 1
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
                    r_tmp += r_q[q] * s_wsp1[k * nq0 * nq1 + q * nq0 + p];
                }
                s_wsp0[p * nm1 * nm2 + j * nm2 + k] = r_tmp;
            }
        }
        __syncthreads();


        //step-11 : direction 0
        if(tid < nm0 * nm1)
        {
            int i = tid / nm1;
            int j = tid % nm1;

            //copy to register
            for(int p = 0; p < nq0; ++p){
                r_p[p] = s_basis0[p * nm0 + i];
            }

            //mat-vec multp
            for(int k = 0; k < nm2; ++k){
                r_tmp = 0;
                for(int p = 0; p < nq0; ++p){
                    r_tmp += r_p[p] * s_wsp0[p * nm1 * nm2 + j * nm2 + k];
                }
                s_wsp1[i * nm1 * nm2 + j * nm2 + k] = r_tmp;
            }
        }
        __syncthreads();


        //step-12 : Copy wsp0 to out
        for(int tidx = threadIdx.x; tidx < nm0 * nm1 * nm2; tidx += blockDim.x)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp1[tid];
        }
        __syncthreads();

        e += gridDim.x;
        }
    }
} //namespace Parallel
} //namespace BK3

#endif //BK3_CUDA_KERNELS_CUH
