#ifndef BK5_CUDA_KERNELS_CUH
#define BK5_CUDA_KERNELS_CUH

#include <timer.hpp>
#include <vector>

namespace BK5{
namespace Parallel{

template<typename T, unsigned int nq0, unsigned int nq1, unsigned int nq2>
__global__ void TransHexKernel_QP_3D_Block_SimpleMap(
    const unsigned int nelmt, const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{

    extern __shared__ T shared[];
    T *s_dbasis0 = shared;
    T *s_dbasis1 = s_dbasis0 + nq0 * nq0;
    T *s_dbasis2 = s_dbasis1 + nq1 * nq1;
    T *rqr     = s_dbasis2 + nq2 * nq2;
    T *rqs     = rqr + nq0 * nq1 * nq2;
    T *rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
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

    T Grr, Grs, Grt, Gss, Gst, Gtt;
    T qr, qs, qt;

    unsigned int e = blockIdx.x;
    while(e < nelmt)
    {   
        const unsigned int tid = threadIdx.x;
        if(tid < nq0 * nq1 * nq2){

            const unsigned int i = tid / (nq1 * nq2);
            const unsigned int j = (tid % (nq1 * nq2)) / nq2;
            const unsigned int k = tid % nq2;

            qr = 0; qs = 0; qt = 0;
    
            //Load Geometric Factors, coalesced access
            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
    
            // Multiply by D
            for(unsigned int n = 0; n < nq0; n++){
                qr += s_dbasis0[i * nq0 + n] * d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }
    
            for(unsigned int n = 0; n < nq1; n++){
                qs += s_dbasis1[j * nq1 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
            }
    
            for(unsigned int n = 0; n < nq2; n++){
                qt += s_dbasis2[k * nq2 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
            }
            
            // Apply chain rule
            rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
            rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
            rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
        
            __syncthreads();

            T tmp0 = 0;
            for(unsigned int n = 0; n < nq0; ++n)
                tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * s_dbasis0[n * nq0 + i];
    
            for(unsigned int n = 0; n < nq1; ++n)                
                tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * s_dbasis1[n * nq1 + j];
    
            for(unsigned int n = 0; n < nq2; ++n)
                tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * s_dbasis2[n * nq2 + k];
    
            d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
        }

        e += gridDim.x;
    }
}


template<typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
__global__ void TransHexKernel_QP_2D_Block_jk_SimpleMap(
    const unsigned int nelmt, const T *__restrict__ d_dbasis0, const T *__restrict__ d_dbasis1,
    const T *__restrict__ d_dbasis2, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    
    T r_i[nq0];
    T r_j[nq1];
    T r_k[nq2];
    
    extern __shared__ T shared[];

    T *s_dbasis0 = shared;
    T *rqr     = s_dbasis0 + nq0 * nq0;
    T *rqs     = rqr + nq0 * nq1 * nq2;
    T *rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nq0; tid += blockDim.x)
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }
    __syncthreads();


    unsigned int e = blockIdx.x;
    while(e < nelmt)
    {   
        for(unsigned int tid = threadIdx.x; tid < nq1 * nq2; tid += blockDim.x){
            unsigned int j = tid / nq2;
            unsigned int k = tid % nq2;

            //copy to register
            for(unsigned int n = 0; n < nq0; n++)
            {
                r_i[n] = d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }

            for(unsigned int n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[j * nq1 + n];
            }

            for(unsigned int n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[k * nq2 + n];
            }

            T Grr, Grs, Grt, Gss, Gst, Gtt;
            T qr, qs, qt;

            for(unsigned int i = 0; i < nq0; ++i){

                qr = 0; qs = 0; qt = 0; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
        
                // Multiply by D
                for(unsigned int n = 0; n < nq0; n++){
                    qr += s_dbasis0[i * nq0 + n] * r_i[n];
                }
        
                for(unsigned int n = 0; n < nq1; n++){
                    qs += r_j[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                }
                
                for(unsigned int n = 0; n < nq2; n++){
                    qt += r_k[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
                }
                
                // Apply chain rule
                rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
            }
            __syncthreads();


            //copy to register
            for(unsigned int n = 0; n < nq0; n++)
            {
                r_i[n] = rqr[n * nq1 * nq2 + j * nq2 + k];
            }

            for(unsigned int n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[n * nq1 + j];
            }

            for(unsigned int n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[n * nq2 + k];
            }

            for(unsigned int i = 0; i < nq0; ++i){
            
                T tmp0 = 0;
                for(unsigned int n = 0; n < nq0; ++n)
                    tmp0 += r_i[n] * s_dbasis0[n * nq0 + i];

                for(unsigned int n = 0; n < nq1; ++n)                
                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * r_j[n];

                for(unsigned int n = 0; n < nq2; ++n)
                    tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * r_k[n];

                d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
                
            }
        }
        e += gridDim.x;
    }
}
} //namespace Parallel
} //namespace BK5

#endif //BK5_CUDA_KERNELS_CUH