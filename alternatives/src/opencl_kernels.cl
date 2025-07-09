#ifndef T
#define T float
#endif

__kernel void BwdTransHexKernel_QP_1D(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, 
    const unsigned int nm1,
    const unsigned int nm2, 
    const unsigned int nelmt, 
    __global const T *restrict d_basis0, 
    __global const T *restrict d_basis1,
    __global const T *restrict d_basis2, 
    __global const T *restrict JxW, 
    __global const T *restrict d_in, 
    __global T *restrict d_out,
    __local T *shared)
{
    __local T *s_basis0 = shared;
    __local T *s_basis1 = s_basis0 + nm0 * nq0;
    __local T *s_basis2 = s_basis1 + nm1 * nq1;
    __local T *s_wsp0 = s_basis2 + nm2 * nq2;
    __local T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //copy to shared memory
    for(unsigned int tid = get_local_id(0); tid < nq0 * nm0; tid += get_local_size(0))
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = get_local_id(0); tid < nq1 * nm1; tid += get_local_size(0))
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = get_local_id(0); tid < nq2 * nm2; tid += get_local_size(0))
    {
        s_basis2[tid] = d_basis2[tid];
    }

    int i, j, k, p, q, r;
    unsigned int e = get_group_id(0);

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = get_local_id(0); tid < nm0 * nm1 * nm2; tid += get_local_size(0))
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-2 : direction 0
        for(unsigned int tid = get_local_id(0); tid < nq0 * nm1 * nm2; tid += get_local_size(0))
        {
            p = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-3 : direction 1
        for(unsigned int tid = get_local_id(0); tid < nq0 * nq1 * nm2; tid += get_local_size(0))
        {
            q = tid / (nq0 * nm2);
            p = (tid % (nq0 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-4 : direction 2
        for(unsigned int tid = get_local_id(0); tid < nq0 * nq1 * nq2; tid += get_local_size(0))
        {
            p = tid / (nq1 * nq2);
            q = (tid % (nq1 * nq2)) / nq2;
            r = tid % nq2;

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = get_local_id(0); tid < nq0 * nq1 * nq2; tid += get_local_size(0)){
            s_wsp1[tid] *= JxW[e * nq0 * nq1 * nq2 + tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-6 : direction 2
        for(unsigned int tid = get_local_id(0); tid < nq0 * nq1 * nm2; tid += get_local_size(0))
        {
            q = tid / (nq0 * nm2);
            p = (tid % (nq0 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-7 : direction 1
        for(unsigned int tid = get_local_id(0); tid < nm1 * nm2 * nq0; tid += get_local_size(0))
        {
            p = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-8 : direction 0
        for(unsigned int tid = get_local_id(0); tid < nm0 * nm1 * nm2; tid += get_local_size(0))
        {
            i = tid / (nm1 * nm2);
            j = (tid % (nm1 * nm2)) / nm2;
            k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-9 : Copy wsp0 to out
        for(unsigned int tid = get_local_id(0); tid < nm0 * nm1 * nm2; tid += get_local_size(0))
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        e += get_num_groups(0);
    }   
}


// In 3D thread-blocks in CUDA, X dimension is fastest, Z is slowest
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy
// get_local_id(0) for i and p
// get_local_id(0) for j and q
// get_local_id(0) for k and r

__kernel void BwdTransHexKernel_QP_1D_3D_BLOCKS(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1,
    const unsigned int nm2, const unsigned int nelmt, 
    __global const T *restrict d_basis0, 
    __global const T *restrict d_basis1,
    __global const T *restrict d_basis2, 
    __global const T *restrict d_JxW, 
    __global const T *restrict d_in, 
    __global T *restrict d_out,
    __local T *shared)
{
    __local T *s_basis0 = shared;
    __local T *s_basis1 = s_basis0 + nm0 * nq0;
    __local T *s_basis2 = s_basis1 + nm1 * nq1;
    __local T *s_wsp0 = s_basis2 + nm2 * nq2;
    __local T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //Finding global indices
    unsigned int blockSize = get_local_size(0) * get_local_size(1) * get_local_size(2);
    unsigned int blockThreadIdx = get_local_id(2) * get_local_size(1) * get_local_size(0)
                                + get_local_id(1) * get_local_size(0) 
                                + get_local_id(0);

    //copy to shared memory
    for(unsigned int tid = blockThreadIdx; tid < nq0 * nm0; tid += blockSize)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = blockThreadIdx; tid < nq1 * nm1; tid += blockSize)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = blockThreadIdx; tid < nq2 * nm2; tid += blockSize)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    
    unsigned int e = get_group_id(0);

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-2 : direction 0
        for(unsigned int p = get_local_id(0); p < nq0; p += get_local_size(0)){
            for(unsigned int j = get_local_id(1); j < nm1; j += get_local_size(1)){
                for(unsigned int k = get_local_id(2); k < nm2; k += get_local_size(2)){

                    T tmp = 0.0;
                    for(unsigned int i = 0; i < nm0; ++i)
                    {
                        tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                    }
                    s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
                    
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-3 : direction 1
        for(unsigned int q = get_local_id(1); q < nq1; q += get_local_size(1)){
            for(unsigned int p = get_local_id(0); p < nq0; p += get_local_size(0)){
                for(unsigned int k = get_local_id(2); k < nm2; k += get_local_size(2)){

                    T tmp = 0.0;
                    for(unsigned int j = 0; j < nm1; j++)
                    {
                        tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                    }
                    s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-4 : direction 2
        for(unsigned int p = get_local_id(0); p < nq0; p += get_local_size(0)){
            for(unsigned int q = get_local_id(1); q < nq1; q += get_local_size(1)){
                for(unsigned int r = get_local_id(2); r < nq2; r += get_local_size(2)){

                    T tmp = 0.0;
                    for(unsigned int k = 0; k < nm2; ++k)
                    {
                        tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                    }
                    s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        

        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = blockThreadIdx; tid < nq0 * nq1 * nq2; tid += blockSize){
            s_wsp1[tid] *= d_JxW[e * nq0 * nq1 * nq2 + tid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-6 : direction 2
        for(unsigned int q = get_local_id(1); q < nq1; q += get_local_size(1)){
            for(unsigned int p = get_local_id(0); p < nq0; p += get_local_size(0)){
                for(unsigned int k = get_local_id(2); k < nm2; k += get_local_size(2)){
                
                    T tmp = 0.0;
                    for(unsigned int r = 0; r < nq2; ++r)
                    {
                        tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
                    }
                    s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-7 : direction 1
        for(unsigned int p = get_local_id(0); p < nq0; p += get_local_size(0)){
            for(unsigned int j = get_local_id(1); j < nm1; j += get_local_size(1)){
                for(unsigned int k = get_local_id(2); k < nm2; k += get_local_size(2)){    

                    T tmp = 0.0;
                    for(unsigned int q = 0; q < nq1; q++)
                    {
                        tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
                    }
                    s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-8 : direction 0
        for(unsigned int i = get_local_id(0); i < nm0; i += get_local_size(0)){
            for(unsigned int j = get_local_id(1); j < nm1; j += get_local_size(1)){
                for(unsigned int k = get_local_id(2); k < nm2; k += get_local_size(2)){

                    T tmp = 0.0;
                    for(unsigned int p = 0; p < nq0; ++p)
                    {
                        tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                    }
                    s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;       
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-9 : Copy wsp0 to out
        for(unsigned int tid = blockThreadIdx; tid < nm0 * nm1 * nm2; tid += blockSize)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        } 
        barrier(CLK_LOCAL_MEM_FENCE);

        e += get_num_groups(0);
    }
}


__kernel void BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2, 
    const unsigned int nelmt, 
    __global const T *restrict d_basis0, 
    __global const T *restrict d_basis1,
    __global const T *restrict d_basis2, 
    __global const T *restrict d_JxW, 
    __global const T *restrict d_in, 
    __global       T *restrict d_out,
    __local T *shared)
{
    __local T *s_basis0 = shared;
    __local T *s_basis1 = s_basis0 + nm0 * nq0;
    __local T *s_basis2 = s_basis1 + nm1 * nq1;
    __local T *s_wsp0 = s_basis2 + nm2 * nq2;
    __local T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //Finding global indices
    unsigned int blockThreadIdx = get_local_id(2) * get_local_size(1) * get_local_size(0) 
                                + get_local_id(1) * get_local_size(0) 
                                + get_local_id(0);
    
    //copy to shared memory
    if(blockThreadIdx < nq0 * nm0)
    {
        s_basis0[blockThreadIdx] = d_basis0[blockThreadIdx];
    }
    
    if(blockThreadIdx < nq1 * nm1)
    {
        s_basis1[blockThreadIdx] = d_basis1[blockThreadIdx];
    }

    if(blockThreadIdx < nq2 * nm2)
    {
        s_basis2[blockThreadIdx] = d_basis2[blockThreadIdx];
    }
    
    
    unsigned int p, q, r, i, j, k;
    unsigned int e = get_group_id(0);
    
    while(e < nelmt)
    {   
        //step-1 : Copy from in to the wsp0
        if(blockThreadIdx < nm0 * nm1 * nm2)
        {
            s_wsp0[blockThreadIdx] = d_in[e * nm0 * nm1 * nm2 + blockThreadIdx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-2 : direction 0
        if(get_local_id(0) < nq0 && get_local_id(1) < nm1 && get_local_id(2) < nm2){
            p = get_local_id(0);
            j = get_local_id(1);
            k = get_local_id(2);

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;   
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-3 : direction 1
        if(get_local_id(1) < nq1 && get_local_id(0) < nq0 && get_local_id(2) < nm2){ 

            q = get_local_id(1);
            p = get_local_id(0);
            k = get_local_id(2);
            
            T tmp = 0.0;
            for(unsigned int j = 0; j < nm1; j++)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        //step-4 : direction 2
        if(get_local_id(0) < nq0 && get_local_id(1) < nq1  && get_local_id(2) < nq2){ 
            p = get_local_id(0);
            q = get_local_id(1);
            r = get_local_id(2);

            T tmp = 0.0;
            for(unsigned int k = 0; k < nm2; ++k)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
            }
            s_wsp1[p * nq1 * nq2 + q * nq2 + r] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //Reverse Operations

        //step-5 : Multiply with weights and determinant of Jacobi
        if(blockThreadIdx < nq0 * nq1 * nq2)
        {
            s_wsp1[blockThreadIdx] *= d_JxW[e * nq0 * nq1 * nq2 + blockThreadIdx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //step-6 : direction 2
        if(get_local_id(0) < nq0 && get_local_id(1) < nq1  && get_local_id(2) < nm2)
        {
            p = get_local_id(0);
            q = get_local_id(1);
            k = get_local_id(2);
        
            T tmp = 0.0;
            for(unsigned int r = 0; r < nq2; ++r)
            {
                tmp += s_wsp1[p * nq1 * nq2 + q * nq2 + r] * s_basis2[r * nm2 + k];
            }
            s_wsp0[q * nq0 * nm2 + p * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-7 : direction 1
        if(get_local_id(0) < nq0 && get_local_id(1) < nm1  && get_local_id(2) < nm2)
        {
            p = get_local_id(0);
            j = get_local_id(1);
            k = get_local_id(2);
        
            T tmp = 0.0;
            for(unsigned int q = 0; q < nq1; q++)
            {
                tmp += s_wsp0[q * nq0 * nm2 + p * nm2 + k]  * s_basis1[q * nm1 + j];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-8 : direction 0
        if(get_local_id(0) < nm0 && get_local_id(1) < nm1 && get_local_id(2) < nm2)
        {
            i = get_local_id(0);
            j = get_local_id(1);
            k = get_local_id(2);
        
            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        //step-9 : Copy wsp0 to out
        if(blockThreadIdx < nm0 * nm1 * nm2)
        {
            d_out[e * nm0 * nm1 * nm2 + blockThreadIdx] = s_wsp0[blockThreadIdx];
        } 
        barrier(CLK_LOCAL_MEM_FENCE);

        e += get_num_groups(0);
    }
}