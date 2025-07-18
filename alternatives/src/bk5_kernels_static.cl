#include "kernels_common.h"

#define BK5_KERNEL_STATIC(name)
__kernel void name( \
    const index_type nelmt, \
    __global const real *restrict d_dbasis0, \
    __global const real *restrict d_dbasis1, \
    __global const real *restrict d_dbasis2, \
    __global const real *restrict d_G, \
    __global const real *restrict d_in, \
    __global       real *restrict d_out, \
    __local        real *restrict shared)


BK5_KERNEL_STATIC(TransHexKernel_QP_3D_Block) 
{

    const index_type nq0 = NQ0;
    const index_type nq1 = NQ1;
    const index_type nq2 = NQ2;

    __local real *restrict s_dbasis0 = shared;
    __local real *restrict s_dbasis1 = s_dbasis0 + nq0 * nq0;
    __local real *restrict s_dbasis2 = s_dbasis1 + nq1 * nq1;
    __local real *restrict rqr       = s_dbasis2 + nq2 * nq2;
    __local real *restrict rqs       = rqr + nq0 * nq1 * nq2;
    __local real *restrict rqt       = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type tid = get_local_id(0); tid < nq0 * nq0; tid += get_local_size(0))
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(index_type tid = get_local_id(0); tid < nq1 * nq1; tid += get_local_size(0))
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(index_type tid = get_local_id(0); tid < nq2 * nq2; tid += get_local_size(0))
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    real Grr, Grs, Grt, Gss, Gst, Gtt;
    real qr, qs, qt;

    index_type e = get_group_id(0);
    while(e < nelmt)
    {   
        for(index_type tid = get_local_id(0); tid < nq0 * nq1 * nq2; tid += get_local_size(0)){

            index_type i = tid / (nq1 * nq2);
            index_type j = (tid % (nq1 * nq2)) / nq2;
            index_type k = tid % nq2;

            qr = 0; qs = 0; qt = 0;
    
            //Load Geometric Factors, coalesced access
            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
    
            // Multiply by D
            for(index_type n = 0; n < nq0; n++){
                qr += s_dbasis0[i * nq0 + n] * d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }
    
            for(index_type n = 0; n < nq1; n++){
                qs += s_dbasis1[j * nq1 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
            }
            
            for(index_type n = 0; n < nq2; n++){
                qt += s_dbasis2[k * nq2 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
            }
            
            // Apply chain rule
            rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
            rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
            rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for(index_type tid = get_local_id(0); tid < nq0 * nq1 * nq2; tid += get_local_size(0)){

            index_type i = tid / (nq1 * nq2);
            index_type j = (tid % (nq1 * nq2)) / nq2;
            index_type k = tid % nq2;

            real tmp0 = ZERO;
            for(index_type n = 0; n < nq0; ++n)
                tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * s_dbasis0[n * nq0 + i];
    
            for(index_type n = 0; n < nq1; ++n)                
                tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * s_dbasis1[n * nq1 + j];
    
            for(index_type n = 0; n < nq2; ++n)
                tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * s_dbasis2[n * nq2 + k];
    
            d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
        }

        e += get_num_groups(0);
    }
}


BK5_KERNEL_STATIC(TransHexKernel_QP_3D_Block_SimpleMap)
{
    const index_type nq0 = NQ0;
    const index_type nq1 = NQ1;
    const index_type nq2 = NQ2;

    __local real *restrict s_dbasis0 = shared;
    __local real *restrict s_dbasis1 = s_dbasis0 + nq0 * nq0;
    __local real *restrict s_dbasis2 = s_dbasis1 + nq1 * nq1;
    __local real *restrict rqr     = s_dbasis2 + nq2 * nq2;
    __local real *restrict rqs     = rqr + nq0 * nq1 * nq2;
    __local real *restrict rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type tid = get_local_id(0); tid < nq0 * nq0; tid += get_local_size(0))
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }

    for(index_type tid = get_local_id(0); tid < nq1 * nq1; tid += get_local_size(0))
    {
        s_dbasis1[tid] = d_dbasis1[tid];
    }

    for(index_type tid = get_local_id(0); tid < nq2 * nq2; tid += get_local_size(0))
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    real Grr, Grs, Grt, Gss, Gst, Gtt;
    real qr, qs, qt;

    index_type e = get_group_id(0);
    while(e < nelmt)
    {   
        const index_type tid = get_local_id(0);
        if(tid < nq0 * nq1 * nq2){

            const index_type i = tid / (nq1 * nq2);
            const index_type j = (tid % (nq1 * nq2)) / nq2;
            const index_type k = tid % nq2;

            qr = 0; qs = 0; qt = 0;
    
            //Load Geometric Factors, coalesced access
            Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
            Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
    
            // Multiply by D
            for(index_type n = 0; n < nq0; n++){
                qr += s_dbasis0[i * nq0 + n] * d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }
    
            for(index_type n = 0; n < nq1; n++){
                qs += s_dbasis1[j * nq1 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
            }
    
            for(index_type n = 0; n < nq2; n++){
                qt += s_dbasis2[k * nq2 + n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
            }
            
            // Apply chain rule
            rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
            rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
            rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
        
            barrier(CLK_LOCAL_MEM_FENCE);

            real tmp0 = ZERO;
            for(index_type n = 0; n < nq0; ++n)
                tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * s_dbasis0[n * nq0 + i];
    
            for(index_type n = 0; n < nq1; ++n)                
                tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * s_dbasis1[n * nq1 + j];
    
            for(index_type n = 0; n < nq2; ++n)
                tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * s_dbasis2[n * nq2 + k];
    
            d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
        }

        e += get_num_groups(0);
    }
}


BK5_KERNEL_STATIC(TransHexKernel_QP_2D_Block_ij) 
{
    const index_type nq0 = NQ0;
    const index_type nq1 = NQ1;
    const index_type nq2 = NQ2;

    real r_i[10];
    real r_j[10];
    real r_k[10];
    
    __local real *restrict s_dbasis2 = shared;
    __local real *restrict rqr     = s_dbasis2 + nq2 * nq2;
    __local real *restrict rqs     = rqr + nq0 * nq1 * nq2;
    __local real *restrict rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type tid = get_local_id(0); tid < nq2 * nq2; tid += get_local_size(0))
    {
        s_dbasis2[tid] = d_dbasis2[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    index_type e = get_group_id(0);
    while(e < nelmt)
    {   
        for(index_type tid = get_local_id(0); tid < nq0 * nq1; tid += get_local_size(0)){
            index_type i = tid / nq1;
            index_type j = tid % nq1;

            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = d_dbasis0[i * nq0 + n];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[j * nq1 + n];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
            }

            real Grr, Grs, Grt, Gss, Gst, Gtt;
            real qr, qs, qt;

            for(index_type k = 0; k < nq2; ++k){

                qr = ZERO; qs = ZERO; qt = ZERO; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
        
                // Multiply by D
                for(index_type n = 0; n < nq0; n++){
                    qr += r_i[n] * d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
                }
        
                for(index_type n = 0; n < nq1; n++){
                    qs += r_j[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                }
                
                for(index_type n = 0; n < nq2; n++){
                    qt += s_dbasis2[k * nq2 + n] * r_k[n];
                }
                
                // Apply chain rule
                rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(index_type tid = get_local_id(0); tid < nq0 * nq1; tid += get_local_size(0)){

            index_type i = tid / nq1;
            index_type j = tid % nq1;

            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = d_dbasis0[n * nq0 + i];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[n * nq1 + j];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = rqt[i * nq1 * nq2 + j * nq2 + n];
            }

            for(index_type k = 0; k < nq2; ++k)
            {
                real tmp0 = ZERO;
                for(index_type n = 0; n < nq0; ++n)
                    tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * r_i[n];
        
                for(index_type n = 0; n < nq1; ++n)                
                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * r_j[n];
        
                for(index_type n = 0; n < nq2; ++n)
                    tmp0 += r_k[n] * s_dbasis2[n * nq2 + k];
        
                d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
            }
        }

        e += get_num_groups(0);
    }
}


BK5_KERNEL_STATIC(TransHexKernel_QP_2D_Block_jk)
{
    const index_type nq0 = NQ0;
    const index_type nq1 = NQ1;
    const index_type nq2 = NQ2;

    real r_i[10];
    real r_j[10];
    real r_k[10];
    
    __local real *restrict s_dbasis0 = shared;
    __local real *restrict rqr     = s_dbasis0 + nq0 * nq0;
    __local real *restrict rqs     = rqr + nq0 * nq1 * nq2;
    __local real *restrict rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type tid = get_local_id(0); tid < nq0 * nq0; tid += get_local_size(0))
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    index_type e = get_group_id(0);
    while(e < nelmt)
    {   
        for(index_type tid = get_local_id(0); tid < nq1 * nq2; tid += get_local_size(0)){
            index_type j = tid / nq2;
            index_type k = tid % nq2;

            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[j * nq1 + n];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[k * nq2 + n];
            }

            real Grr, Grs, Grt, Gss, Gst, Gtt;
            real qr, qs, qt;

            for(index_type i = 0; i < nq0; ++i){

                qr = ZERO; qs = ZERO; qt = ZERO; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
        
                // Multiply by D
                for(index_type n = 0; n < nq0; n++){
                    qr += s_dbasis0[i * nq0 + n] * r_i[n];
                }
        
                for(index_type n = 0; n < nq1; n++){
                    qs += r_j[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                }
                
                for(index_type n = 0; n < nq2; n++){
                    qt += r_k[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
                }
                
                // Apply chain rule
                rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(index_type tid = get_local_id(0); tid < nq1 * nq2; tid += get_local_size(0)){

            index_type j = tid / nq2;
            index_type k = tid % nq2;

            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = rqr[n * nq1 * nq2 + j * nq2 + k];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[n * nq1 + j];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[n * nq2 + k];
            }

            for(index_type i = 0; i < nq0; ++i)
            {
                real tmp0 = ZERO;
                for(index_type n = 0; n < nq0; ++n)
                    tmp0 += r_i[n] * s_dbasis0[n * nq0 + i];
        
                for(index_type n = 0; n < nq1; ++n)                
                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * r_j[n];
        
                for(index_type n = 0; n < nq2; ++n)
                    tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * r_k[n];
        
                d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
            }
        }

        e += get_num_groups(0);
    }
}


BK5_KERNEL_STATIC(TransHexKernel_QP_2D_Block_jk_SimpleMap)
{
    const index_type nq0 = NQ0;
    const index_type nq1 = NQ1;
    const index_type nq2 = NQ2;

    real r_i[10];
    real r_j[10];
    real r_k[10];
    
    __local real *restrict s_dbasis0 = shared;
    __local real *restrict rqr     = s_dbasis0 + nq0 * nq0;
    __local real *restrict rqs     = rqr + nq0 * nq1 * nq2;
    __local real *restrict rqt     = rqs + nq0 * nq1 * nq2;

    //copy to shared memory
    for(index_type tid = get_local_id(0); tid < nq0 * nq0; tid += get_local_size(0))
    {
        s_dbasis0[tid] = d_dbasis0[tid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    index_type e = get_group_id(0);
    while(e < nelmt)
    {   
        for(index_type tid = get_local_id(0); tid < nq1 * nq2; tid += get_local_size(0)){
            index_type j = tid / nq2;
            index_type k = tid % nq2;

            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = d_in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[j * nq1 + n];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[k * nq2 + n];
            }

            real Grr, Grs, Grt, Gss, Gst, Gtt;
            real qr, qs, qt;

            for(index_type i = 0; i < nq0; ++i){

                qr = ZERO; qs = ZERO; qt = ZERO; 
    
                //Load Geometric Factors, coalesced access
                Grr = d_G[e * 6 * nq0 * nq1 * nq2 + 0 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grs = d_G[e * 6 * nq0 * nq1 * nq2 + 1 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Grt = d_G[e * 6 * nq0 * nq1 * nq2 + 2 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gss = d_G[e * 6 * nq0 * nq1 * nq2 + 3 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gst = d_G[e * 6 * nq0 * nq1 * nq2 + 4 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
                Gtt = d_G[e * 6 * nq0 * nq1 * nq2 + 5 * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k];
        
                // Multiply by D
                for(index_type n = 0; n < nq0; n++){
                    qr += s_dbasis0[i * nq0 + n] * r_i[n];
                }
        
                for(index_type n = 0; n < nq1; n++){
                    qs += r_j[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k];
                }
                
                for(index_type n = 0; n < nq2; n++){
                    qt += r_k[n] * d_in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n];
                }
                
                // Apply chain rule
                rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
            }
            barrier(CLK_LOCAL_MEM_FENCE);


            //copy to register
            for(index_type n = 0; n < nq0; n++)
            {
                r_i[n] = rqr[n * nq1 * nq2 + j * nq2 + k];
            }

            for(index_type n = 0; n < nq1; n++)
            {
                r_j[n] = d_dbasis1[n * nq1 + j];
            }

            for(index_type n = 0; n < nq2; n++)
            {
                r_k[n] = d_dbasis2[n * nq2 + k];
            }

            for(index_type i = 0; i < nq0; ++i){
            
                real tmp0 = ZERO;
                for(index_type n = 0; n < nq0; ++n)
                    tmp0 += r_i[n] * s_dbasis0[n * nq0 + i];

                for(index_type n = 0; n < nq1; ++n)                
                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * r_j[n];

                for(index_type n = 0; n < nq2; ++n)
                    tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * r_k[n];

                d_out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
                
            }
        }
        e += get_num_groups(0);
    }
}