#ifndef BK3_TEMPLATED_CUDA_KERNELS_CUH
#define BK3_TEMPLATED_CUDA_KERNELS_CUH

#include <timer.hpp>
#include <vector>

namespace BK3{
namespace Parallel{


template<typename T, const unsigned int nq>
__global__ void LaplaceOperator(
    size_t nelmt, const unsigned int nelmtPerBatch, const T*__restrict__ d_basis, const T *__restrict__ d_dbasis, 
    const T *__restrict__ d_G, const T *__restrict__ d_in, T *__restrict__ d_out)
{
    const unsigned int nm = nq - 1;

    T r_p[nq];
    T r_q[nq];
    T r_r[nq];

    extern __shared__ T shared[];
    T *s_basis  = shared;
    T *s_dbasis = s_basis  + nq * nm;

    T *s_wsp0    = s_dbasis + nq * nq;
    T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;

    T *s_rqr     = s_wsp1   + nelmtPerBatch * nq * nq * nq;
    T *s_rqs     = s_rqr    + nelmtPerBatch * nq * nq * nq;
    T *s_rqt     = s_wsp0;


    //copy to shared memory
    for(int tid = threadIdx.x; tid < nm * nq; tid += blockDim.x )
    {
        s_basis[tid] = d_basis[tid];
    }


    for(int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x)
    {
        s_dbasis[tid] = d_dbasis[tid];
    }
    __syncthreads();

    /*
    Interpolate to GL nodes
    */

    //element batch iteration
    size_t eb = blockIdx.x;
    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
    {   
        //current nelmtPerBatch (edge case, last batch size can be less)
        int remaining = nelmt - eb * nelmtPerBatch;
        int c_nelmtPerBatch = (nelmtPerBatch < remaining) ? nelmtPerBatch : remaining;
        
        //step-1 : Copy from in to the wsp0
        for(int i = threadIdx.x; i < c_nelmtPerBatch * nm * nm * nm; i += blockDim.x)
        {
            s_wsp0[i] = d_in[eb * nelmtPerBatch * nm * nm * nm + i];
        }
        __syncthreads();

        //step-2 : direction 0
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm; tid += blockDim.x)
            {
                int e = tid / (nm * nm);
                int j = tid % (nm * nm) / nm;
                int k = tid % nm;

                for(int i=0; i<nm; ++i){
                    r_p[i] = s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k];
                }

                for (int p = 0; p < nq; ++p) {
                    T tmp = 0.0;
                
                    for(int i = 0; i < nm; ++i) {
                        tmp += s_basis[i * nq + p] * r_p[i];
                    }
                
                    s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                }
            }
            __syncthreads();


        //step-3 : direction 1
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nq; tid += blockDim.x)
        {
            int e = tid / (nq * nm);
            int p = tid % (nq * nm) / nm;
            int k = tid % nm;
            for(int j=0; j<nm; ++j){
                r_q[j] = s_wsp1[e * nq*nm*nm + p * nm*nm + j * nm + k];
            }

            for (int q = 0; q < nq; ++q) {
                T tmp = 0.0;

                for(int j = 0; j < nm; ++j) {
                    tmp += s_basis[j * nq + q] * r_q[j];
                }

                s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
            }
        }
        __syncthreads();


        //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x)
        {
            int e = tid / (nq * nq);
            int q = tid % (nq * nq) / nq;
            int p = tid % nq;

            for(int k=0; k<nm; ++k){
                r_r[k] = s_wsp0[e * nq*nq*nm + q * nq*nm + p * nm + k];
            }
            for (int r = 0; r < nq; ++r) {
                T tmp = 0.0;

                for(int k = 0; k < nm; ++k) {
                    tmp += s_basis[k * nq + r] * r_r[k];
                }

                s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp;
            }
        }
        __syncthreads();

        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x){
            
            int e = tid / (nq * nq);
            int q = tid % (nq * nq) / nq;
            int r = tid % nq;

            //copy to register
            for(int n = 0; n < nq; n++)
            {
                r_p[n] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + n];
                r_q[n] = s_dbasis[n * nq + q];
                r_r[n] = s_dbasis[n * nq + r];
            }

            T Grr, Grs, Grt, Gss, Gst, Gtt;
            T qr, qs, qt;

            for(int p = 0; p < nq; ++p){

                qr = 0; qs = 0; qt = 0; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 0 * nq*nq*nq + p * nq * nq + q * nq + r];
                Grs = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 1 * nq*nq*nq + p * nq * nq + q * nq + r];
                Grt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 2 * nq*nq*nq + p * nq * nq + q * nq + r];
                Gss = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 3 * nq*nq*nq + p * nq * nq + q * nq + r];
                Gst = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 4 * nq*nq*nq + p * nq * nq + q * nq + r];
                Gtt = d_G[eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq + 5 * nq*nq*nq + p * nq * nq + q * nq + r];
        
                // Multiply by D
                for(int n = 0; n < nq; n++){
                    qr += s_dbasis[n * nq + p] * r_p[n];
                    qs += r_q[n] * s_wsp1[e * nq*nq*nq + r * nq*nq + n * nq + p];
                    qt += r_r[n] * s_wsp1[e * nq*nq*nq + n * nq*nq + q * nq + p];
                }

                // Apply chain rule
                s_rqr[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grr * qt + Grs * qs + Grt * qr;
                s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grs * qt + Gss * qs + Gst * qr;
                s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grt * qt + Gst * qs + Gtt * qr;
            }
        }
        __syncthreads();

        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x){

            int e = tid / (nq * nq);
            int q = tid % (nq * nq) / nq;
            int r = tid % nq;

            //copy to register
            for(int n = 0; n < nq; n++)
            {
                r_p[n] = s_rqr[e * nq*nq*nq + n * nq * nq + q * nq + r];
                r_q[n] = s_dbasis[q * nq + n];
                r_r[n] = s_dbasis[r * nq + n];
            }

            for(int p = 0; p < nq; ++p)
            {
                T tmp0 = 0;
                for(int n = 0; n < nq; ++n)
                    tmp0 += r_p[n] * s_dbasis[p * nq + n];
        
                for(int n = 0; n < nq; ++n)                
                    tmp0 += s_rqs[e * nq*nq*nq + p * nq * nq + n * nq + r] * r_q[n];
        
                for(int n = 0; n < nq; ++n)
                    tmp0 += s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + n] * r_r[n];
        
                s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
            }
        }
        __syncthreads();


        /*
        Interpolate to GLL nodes
        */
        
        //step-9 : direction 2
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x)
            {                
                int e = tid / (nq * nq);
                int q = tid % (nq * nq) / nq;
                int p = tid % nq;

                for(int r=0; r<nq; ++r){
                    r_r[r] = s_wsp1[e * nq*nq*nq + r * nq*nq + q * nq + p];
                }

                for (int k = 0; k < nm; ++k) {
                    T tmp = 0.0;

                    for(int r = 0; r < nq; ++r) {
                        tmp += s_basis[k * nq + r] * r_r[r];
                    }

                    s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                }   
            }
        __syncthreads();


        //step-10 : direction 1
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nq; tid += blockDim.x)
            {   
                int e = tid / (nm * nq);
                int k = tid % (nm * nq) / nq;
                int p = tid % nq;

                for(int q=0; q<nq; ++q){
                    r_q[q] = s_wsp0[e * nm*nq*nq + k * nq*nq + q * nq + p];
                }

                for (int j = 0; j < nm; ++j) {
                    T tmp = 0.0;

                    for(int q = 0; q < nq; ++q) {
                        tmp += s_basis[j * nq + q] * r_q[q];
                    }
                    s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                }
            }
        __syncthreads();


        //step-11 : direction 0
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm; tid += blockDim.x)
            {   
                int e = tid / (nm * nm);
                int j = tid % (nm * nm) / nm;
                int k = tid % nm;

                for(int p=0; p<nq; ++p){
                    r_p[p] = s_wsp1[e * nm*nm*nq + k * nm*nq + j * nq + p];
                }

                for (int i = 0; i < nm; ++i) {
                    T tmp = 0.0;
                    for(int p = 0; p < nq; ++p) {
                        tmp += s_basis[i * nq + p] * r_p[p];
                    }
                    s_wsp0[e * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                }
            }
        __syncthreads();


        //step-12 : Copy wsp0 to out
        for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm * nm; tid += blockDim.x)
        {
            d_out[eb * nelmtPerBatch * nm * nm * nm + tid] = s_wsp0[tid];
        }
        __syncthreads();

        eb += gridDim.x;
    }
}

} //namespace Parallel
} //namespace BK3

#endif //BK3_TEMPLATED_CUDA_KERNELS_CUH

