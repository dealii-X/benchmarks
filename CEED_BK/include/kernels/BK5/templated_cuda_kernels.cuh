#ifndef BK5_TEMPLATED_CUDA_KERNELS_CUH
#define BK5_TEMPLATED_CUDA_KERNELS_CUH

#include <timer.hpp>
#include <vector>

namespace BK5{
namespace Parallel{


template<typename T, const unsigned int nq>
__global__ void LaplaceOperator(
    const unsigned int nelmt, const unsigned int nelmtPerBatch,
    const T *__restrict__ d_dbasis, const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    
    T r_i[nq];
    T r_j[nq];
    T r_k[nq];
    
    extern __shared__ T shared[];

    T *s_dbasis = shared;
    T *s_rqr     = s_dbasis + nq * nq;
    T *s_rqs     = s_rqr + nelmtPerBatch * nq * nq * nq;
    T *s_rqt     = s_rqs + nelmtPerBatch * nq * nq * nq;
    T *s_wsp      = s_rqt + nelmtPerBatch * nq * nq * nq;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x)
    {
        s_dbasis[tid] = d_dbasis[tid];
    }
    __syncthreads();


    //element batch iteration
    int eb = blockIdx.x;
    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
    {   
        //current nelmtPerBatch (edge case, last batch size can be less)
        int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ? (nelmt - eb * nelmtPerBatch) : nelmtPerBatch;

        for(unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq * nq; tid += blockDim.x)
        {
            s_wsp[tid] = d_in[eb * nelmtPerBatch * nq * nq * nq + tid];
        }
        __syncthreads();

        for(unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x){
            
            int b = tid / (nq * nq);
            int j = tid % (nq * nq) / nq;
            int k = tid % nq;

            //copy to register
            for(unsigned int n = 0; n < nq; n++)
            {
                r_i[n] = s_wsp[b * nq*nq*nq + n * nq*nq + j * nq + k];
                r_j[n] = s_dbasis[j * nq + n];
                r_k[n] = s_dbasis[k * nq + n];
            }

            T Grr, Grs, Grt, Gss, Gst, Gtt;
            T qr, qs, qt;

            for(unsigned int i = 0; i < nq; ++i){

                qr = 0; qs = 0; qt = 0; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 0 * nq*nq*nq + i * nq * nq + j * nq + k];
                Grs = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 1 * nq*nq*nq + i * nq * nq + j * nq + k];
                Grt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 2 * nq*nq*nq + i * nq * nq + j * nq + k];
                Gss = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 3 * nq*nq*nq + i * nq * nq + j * nq + k];
                Gst = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 4 * nq*nq*nq + i * nq * nq + j * nq + k];
                Gtt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + b * 6 * nq*nq*nq + 5 * nq*nq*nq + i * nq * nq + j * nq + k];
        
                // Multiply by D
                for(unsigned int n = 0; n < nq; n++){
                    qr += s_dbasis[i * nq + n] * r_i[n];
                    qs += r_j[n] * s_wsp[b * nq*nq*nq + i * nq * nq + n * nq + k];
                    qt += r_k[n] * s_wsp[b * nq*nq*nq + i * nq * nq + j * nq + n];
                }

                // Apply chain rule
                s_rqr[b * nq*nq*nq + i * nq * nq + j * nq + k] = Grr * qr + Grs * qs + Grt * qt;
                s_rqs[b * nq*nq*nq + i * nq * nq + j * nq + k] = Grs * qr + Gss * qs + Gst * qt;
                s_rqt[b * nq*nq*nq + i * nq * nq + j * nq + k] = Grt * qr + Gst * qs + Gtt * qt;
            }
        }
        __syncthreads();

        for(unsigned int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x){

            int b = tid / (nq * nq);
            int j = tid % (nq * nq) / nq;
            int k = tid % nq;

            //copy to register
            for(unsigned int n = 0; n < nq; n++)
            {
                r_i[n] = s_rqr[b * nq*nq*nq + n * nq * nq + j * nq + k];
                r_j[n] = s_dbasis[n * nq + j];
                r_k[n] = s_dbasis[n * nq + k];
            }

            for(unsigned int i = 0; i < nq; ++i)
            {
                T tmp0 = 0;
                for(unsigned int n = 0; n < nq; ++n)
                    tmp0 += r_i[n] * s_dbasis[n * nq + i];
        
                for(unsigned int n = 0; n < nq; ++n)                
                    tmp0 += s_rqs[b * nq*nq*nq + i * nq * nq + n * nq + k] * r_j[n];
        
                for(unsigned int n = 0; n < nq; ++n)
                    tmp0 += s_rqt[b * nq*nq*nq + i * nq * nq + j * nq + n] * r_k[n];
        
                d_out[eb * nelmtPerBatch * nq * nq * nq + b  * nq * nq * nq + i * nq * nq + j * nq + k] = tmp0;
            }
        }

        eb += gridDim.x;
    }
}



} //namespace Parallel
} //namespace BK5

#endif //BK5_TEMPLATED_CUDA_KERNELS_CUH