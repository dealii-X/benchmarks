#ifndef BK1_TEMPLATED_CUDA_KERNELS_CUH
#define BK1_TEMPLATED_CUDA_KERNELS_CUH

#include <timer.hpp>
#include <vector>

namespace BK1{
namespace Parallel{

template <typename T, const unsigned int nq>
    __global__ void MassOperator(
        const unsigned int nelmt, const unsigned int nelmtPerBatch, const T *__restrict__ d_basis,
        const T* __restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out)
    {
        const unsigned int nm = nq - 1;

        T reg[nq];
        
        extern __shared__ T shared[];
        T *s_basis = shared;
        T *s_wsp0 = s_basis + nm * nq;
        T *s_wsp1 = s_wsp0 + nelmtPerBatch * nq * nq * nq;
        
        //copy to shared memory
        for(unsigned int i = threadIdx.x; i < nm * nq; i += blockDim.x )
        {
            s_basis[i] = d_basis[i];
        }
        __syncthreads();

        //element batch iteration
        int eb = blockIdx.x;
        while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
        {   
            //current nelmtPerBatch (edge case, last batch size can be less)
            int c_nelmtPerBatch = (eb * nelmtPerBatch + nelmtPerBatch > nelmt) ? (nelmt - eb * nelmtPerBatch) : nelmtPerBatch; 
            
            
            //step-1 : Copy from in to the wsp0
            for(int i = threadIdx.x; i < c_nelmtPerBatch * nm * nm * nm; i += blockDim.x)
            {
                s_wsp0[i] = d_in[eb * nelmtPerBatch * nm * nm * nm + i];
            }
            __syncthreads();

            //step-2 : direction 0
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm; tid += blockDim.x)
            {
                int b = tid / (nm * nm);
                int j = tid % (nm * nm) / nm;
                int k = tid % nm;

                for(int i=0; i<nm; ++i){
                    reg[i] = s_wsp0[b * nm*nm*nm + i * nm*nm + j * nm + k];
                }

                for (int p = 0; p < nq; ++p) {
                    T tmp = 0.0;
                
                    for(int i = 0; i < nm; ++i) {
                        tmp += s_basis[p * nm + i] * reg[i];
                    }
                
                    s_wsp1[b * nq*nm*nm + p * nm*nm + j * nm + k] = tmp;
                }
            }
            __syncthreads();


            //step-3 : direction 1
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nq; tid += blockDim.x)
            {
                int b = tid / (nq * nm);
                int p = tid % (nq * nm) / nm;
                int k = tid % nm;

                for(int j=0; j<nm; ++j){
                    reg[j] = s_wsp1[b * nq*nm*nm + p * nm*nm + j * nm + k];
                }

                for (int q = 0; q < nq; ++q) {
                    T tmp = 0.0;

                    for(int j = 0; j < nm; ++j) {
                        tmp += s_basis[q * nm + j] * reg[j];
                    }

                    s_wsp0[b * nq*nq*nm + q * nq*nm + p * nm + k] = tmp;
                }
            }
            __syncthreads();


            //step-4 : direction 2 + step-5 : Multiply with weights and determinant of Jacobi
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x)
            {
                int b = tid / (nq * nq);
                int q = tid % (nq * nq) / nq;
                int p = tid % nq;

                for(int k=0; k<nm; ++k){
                    reg[k] = s_wsp0[b * nq*nq*nm + q * nq*nm + p * nm + k];
                }
                for (int r = 0; r < nq; ++r) {
                    T tmp = 0.0;

                    for(int k = 0; k < nm; ++k) {
                        tmp += s_basis[r * nm + k] * reg[k];
                    }

                    s_wsp1[b * nq*nq*nq + r * nq*nq + q * nq + p] = tmp * d_JxW[eb * nelmtPerBatch * nq * nq * nq + b * nq*nq*nq + r * nq*nq + q * nq + p];
                }
            }
            __syncthreads();

            //Reverse Operations
    
            //step-6 : direction 2
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nq * nq; tid += blockDim.x)
            {                
                int b = tid / (nq * nq);
                int q = tid % (nq * nq) / nq;
                int p = tid % nq;

                for(int r=0; r<nq; ++r){
                    reg[r] = s_wsp1[b * nq*nq*nq + r * nq*nq + q * nq + p];
                }

                for (int k = 0; k < nm; ++k) {
                    T tmp = 0.0;

                    for(int r = 0; r < nq; ++r) {
                        tmp += s_basis[r * nm + k] * reg[r];
                    }

                    s_wsp0[b * nm*nq*nq + k * nq*nq + q * nq + p] = tmp;
                }   
            }
            __syncthreads();


            //step-7 : direction 1
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nq; tid += blockDim.x)
            {   
                int b = tid / (nm * nq);
                int k = tid % (nm * nq) / nq;
                int p = tid % nq;

                for(int q=0; q<nq; ++q){
                    reg[q] = s_wsp0[b * nm*nq*nq + k * nq*nq + q * nq + p];
                }

                for (int j = 0; j < nm; ++j) {
                    T tmp = 0.0;

                    for(int q = 0; q < nq; ++q) {
                        tmp += s_basis[q * nm + j] * reg[q];
                    }
                    s_wsp1[b * nm*nm*nq + k * nm*nq + j * nq + p] = tmp;
                }
            }
            __syncthreads();

            //step-8 : direction 0
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm; tid += blockDim.x)
            {   
                int b = tid / (nm * nm);
                int j = tid % (nm * nm) / nm;
                int k = tid % nm;

                for(int p=0; p<nq; ++p){
                    reg[p] = s_wsp1[b * nm*nm*nq + k * nm*nq + j * nq + p];
                }

                for (int i = 0; i < nm; ++i) {
                    T tmp = 0.0;
                    for(int p = 0; p < nq; ++p) {
                        tmp += s_basis[p * nm + i] * reg[p];
                    }
                    s_wsp0[b * nm*nm*nm + i * nm*nm + j * nm + k] = tmp;
                }
            }
            __syncthreads();


            //step-9 : Copy wsp0 to out
            for(int tid = threadIdx.x; tid < c_nelmtPerBatch * nm * nm * nm; tid += blockDim.x)
            {
                d_out[eb * nelmtPerBatch * nm * nm * nm + tid] = s_wsp0[tid];
            }
            __syncthreads();

            eb += gridDim.x;
        }
    }   

} //namespace Parallel
} //namespace BK1

#endif //BK1_TEMPLATED_CUDA_KERNELS_CUH