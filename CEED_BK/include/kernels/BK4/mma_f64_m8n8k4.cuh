#ifndef MMA_F64_M8N8K4_MASS_CUH
#define MMA_F64_M8N8K4_MASS_CUH


namespace Parallel {

template<int M, int N, int K, typename FetchA, typename FetchB, typename StoreC>
__device__ void f64_m8n8k4_tiled_gemm(FetchA get_A, FetchB get_B, StoreC set_C)
{   
    constexpr int m = 8;
    constexpr int n = 8;
    constexpr int k = 4;

    const int tid = threadIdx.x;
    const int laneid = tid % warpSize;

    constexpr int num_tiles_m = (M + m - 1) / m;
    constexpr int num_tiles_n = (N + n - 1) / n;
    constexpr int num_tiles_k = (K + k - 1) / k;

    double r_b[num_tiles_k][num_tiles_n] = {0.0};

    // 1. Copy Matrix B from shared memory to registers via Lambda
    {
        int base_row = laneid % 4;
        int base_col = laneid >> 2;
        
        #pragma unroll
        for(int i = 0; i < num_tiles_k; i++){
            int row = base_row + i * k;

            #pragma unroll
            for(int j = 0; j < num_tiles_n; j++){
                int col = base_col + j * n;
                if(row < K && col < N){
                    r_b[i][j] = get_B(row, col);
                } 
                else{
                    r_b[i][j] = 0.0;
                }
            }
        }
        __syncwarp();
    }

    double r_a[num_tiles_m][num_tiles_k] = {0.0};
    double r_c[num_tiles_m][num_tiles_n][2] = {0.0};
        
    // 2. Copy Matrix A from shared memory to registers via Lambda
    {
        const int base_row = laneid >> 2;
        const int base_col = laneid % 4;

        #pragma unroll
        for(int i = 0; i < num_tiles_m; i++){
            int row = base_row + i * m;
    
            #pragma unroll
            for(int j = 0; j < num_tiles_k; j++){
                int col = base_col + j * k;
                if(row < M && col < K){
                    r_a[i][j] = get_A(row, col);
                }
                else{
                    r_a[i][j] = 0.0;
                }
            }
        }
        __syncwarp();
    }

    // 3. Tiled Tensor Core MMA Computation
    #pragma unroll
    for(int i = 0; i < num_tiles_m; i++){
        #pragma unroll
        for(int j = 0; j < num_tiles_n; j++){
            #pragma unroll
            for(int t = 0; t < num_tiles_k; t++)
            {
                asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0, %1}, {%2}, {%3}, {%0, %1}; \n"
                    : "+d"(r_c[i][j][0]), "+d"(r_c[i][j][1])
                    : "d"(r_a[i][t]),
                      "d"(r_b[t][j])
                );
            }
        }
    }

    // 4. Copy accumulator results to destination memory via Lambda
    {
        const int base_row = laneid >> 2;
        const int base_col = (laneid % 4) * 2;
        
        #pragma unroll
        for(int i = 0; i < num_tiles_m; ++i){
            int row = base_row + i * m;
            #pragma unroll
            for(int j = 0; j < num_tiles_n; ++j){
                int col = base_col + j * n;
                
                if(row < M && col < N){
                    set_C(row, col) = r_c[i][j][0];
                }
                
                int col_next = col + 1;
                if(row < M && col_next < N){
                    set_C(row, col_next) = r_c[i][j][1];
                }
            }
        }
        __syncwarp();
    }
}


template<const unsigned int nq, const unsigned int nm, const unsigned int nelmtPerBatch>
void __global__ f64_m8n8k4_mma(
    const unsigned int nelmt,
    const double *__restrict__ d_basis, const double *__restrict__ d_dbasis,
    const double *__restrict__ d_G, double *__restrict__ d_in, double * __restrict__ d_out, const unsigned int ntests) 
{
    using T = double;

    // Total DoFs per element for a Raviart-Thomas vector
    constexpr unsigned int ndof_1D = nm * nm * nm;

    extern __shared__ T scratch[];
    T *s_basis    = scratch;
    T *s_dbasis = s_basis  + nq * nm;

    T *s_wsp0    = s_dbasis + nq * nq;
    T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;

    T *s_rqr     = s_wsp1   + nelmtPerBatch * nq * nq * nq;
    T *s_rqs     = s_rqr    + nelmtPerBatch * nq * nq * nq;
    T *s_rqt     = s_wsp1;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nm * nq; tid += blockDim.x)
    {
        s_basis[tid] = d_basis[tid];
    }


    for(unsigned int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x)
    {
        s_dbasis[tid] = d_dbasis[tid];
    }
    __syncwarp();



    //element batch iteration
    unsigned int eb = blockIdx.x;

    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
    {

        for(int c = 0; c < 3; ++c)
        {
        const size_t comp_offset  = c * nelmt * nm * nm * nm;
        const size_t batch_offset = eb * nelmtPerBatch * nm * nm * nm;

        //step-1 : Copy from in to the wsp0
        for(int i = threadIdx.x; i < nelmtPerBatch * nm * nm * nm; i += blockDim.x)
        {
            s_wsp1[i] = d_in[comp_offset + batch_offset + i];
        }
        __syncwarp();


        // ==========================================
        // PHASE 1: Interpolate to Quadrature Nodes
        // ==========================================

        // --- Component 0 (x-direction) ---

        //s_wsp1(eijk) . s_basis(kr) = s_wsp0(eijr)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e   = row / (nm * nm);
                const int i   = (row / nm) % nm;
                const int j   = row % nm;

                const int k = col;

                return s_wsp1[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];
            };

            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                return s_basis[row * nq + col];
            };

            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nm * nm);
                const int i = (row / nm) % nm;
                const int j = row % nm;

                const int r = col;

                return s_wsp0[e * (nm * nm * nq) + i * (nm * nq) + j * nq + r];
            };

            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nm, nq, nm>(v_s_wsp1, v_s_basis, v_s_wsp0);
        }


        //s_wsp0(eijr) . s_basis(jq) = s_wsp1(eirq)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nm * nq);
                const int i = (row / nq) % nm;
                const int r = row % nq;

                const int j = col;
            
                return s_wsp0[e * (nm * nm * nq) + i * (nm * nq) + j * nq + r];
            };

            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                const int j = row;
                const int q = col;
            
                return s_basis[j * nq + q];
            };

            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nm * nq);
                const int i = (row / nq) % nm;
                const int r = row % nq;

                const int q = col;
            
                return s_wsp1[e * (nm * nq * nq) + i * (nq * nq) + r * nq + q];
            };

            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nq, nq, nm>(v_s_wsp0, v_s_basis, v_s_wsp1);
        }

        //s_wsp1(eirq) . s_basis(ip) = s_wsp0(erqp)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int q = row % nq;

                const int i = col;
                    
                return s_wsp1[e * (nm * nq * nq) + i * (nq * nq) + r * nq + q];
            };
        
            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                const int i = row;
                const int p = col;
            
                return s_basis[i * nq + p];
            };
        
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int q = row % nq;

                const int p = col;
            
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };
        
            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nm>(v_s_wsp1, v_s_basis, v_s_wsp0);
        }


        // ==========================================
        // PHASE 2: Apply Grad on Quad. Pts.
        // ==========================================
        
        //s_wsp0(erqp) . s_dbasis(ip) = s_rqr(eiqr)
{
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int q = row % nq;

                const int p = col;
            
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };
        
            auto v_s_dbasis = [=] __device__ (const int row, const int col) -> double& {
                const int p = col;
                const int i = row;

                return s_dbasis[i * nq + p];
            };

            auto v_s_rqr = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq; 
                const int q = row % nq;        
                
                const int i = col;             
            
                return s_rqr[e * (nq * nq * nq) + i * (nq * nq) + q * nq + r];
            };
        
            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nq>(v_s_wsp0, v_s_dbasis, v_s_rqr);
        }

        
        //s_wsp0(erqp) . s_dbasis(jq) = s_rqs(epjr)
        {
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int p = row % nq;

                const int q = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };

            auto v_s_dbasis = [=] __device__ (const int row, const int col) -> double& {
                const int q = col;
                const int j = row;

                return s_dbasis[j * nq + q];
            };

            auto v_s_rqs = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq; 
                const int p = row % nq;        

                const int j = col;             
                return s_rqs[e * (nq * nq * nq) + p * (nq * nq) + j * nq + r];
            };

            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nq>(v_s_wsp0, v_s_dbasis, v_s_rqs);
        }

        //s_wsp0(erqp) . s_dbasis(kr) = s_rqt(epqk)
        {
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int p = (row / nq) % nq;
                const int q = row % nq;

                const int r = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };

            auto v_s_dbasis = [=] __device__ (const int row, const int col) -> double& {
                const int r = col;
                const int k = row;
                return s_dbasis[k * nq + r];
            };

            auto v_s_rqt = [=] __device__ (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int p = (row / nq) % nq; 
                const int q = row % nq;        

                const int k = col;             

                return s_rqt[e * (nq * nq * nq) + p * (nq * nq) + q * nq + k];
            };

            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nq>(v_s_wsp0, v_s_dbasis, v_s_rqt);
        }
        

        
        // ==========================================
        // PHASE 3: Apply G
        // ==========================================

        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x) {
            const int e = tid / (nq * nq);
            const int q = (tid / nq) % nq;
            const int r = tid % nq;

            T r_p[nq], r_q[nq], r_r[nq];

            for(int p = 0; p < nq; ++p) {
                const size_t idx = e * (nq * nq * nq) + p * (nq * nq) + q * nq + r;
                r_p[p] = s_rqr[idx];
                r_q[p] = s_rqs[idx];
                r_r[p] = s_rqt[idx];
            }

            for(int p = 0; p < nq; ++p) {
                const size_t g_base = eb * nelmtPerBatch * 6 * nq * nq * nq + e * 6 * nq * nq * nq + p * nq * nq + q * nq + r;
                
                const T Grr = d_G[g_base + 0 * nq * nq * nq];
                const T Grs = d_G[g_base + 1 * nq * nq * nq];
                const T Grt = d_G[g_base + 2 * nq * nq * nq];
                const T Gss = d_G[g_base + 3 * nq * nq * nq];
                const T Gst = d_G[g_base + 4 * nq * nq * nq];
                const T Gtt = d_G[g_base + 5 * nq * nq * nq];

                const T qr = r_p[p];
                const T qs = r_q[p];
                const T qt = r_r[p];

                const size_t idx = e * (nq * nq * nq) + p * (nq * nq) + q * nq + r;
                
                s_rqr[idx] = Grr * qt + Grs * qs + Grt * qr;
                s_rqs[idx] = Grs * qt + Gss * qs + Gst * qr;
                s_rqt[idx] = Grt * qt + Gst * qs + Gtt * qr;
            }
        }
        __syncwarp();

        //Divergence

        //s_rqr(epqr) -> s_rqr(eqrp)
        //s_rqr * d_basis + s_rqs * d_basis + s_rqt * d_basis = s_wsp0


        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
        
            int e = tid / (nq * nq);
            int q = tid % (nq * nq) / nq;
            int r = tid % nq;
            
            T r_p[nq], r_q[nq], r_r[nq];

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

                s_wsp0[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
            }
        }
        __syncwarp();

        
        // ==========================================
        // PHASE 4: Project back to Nodes
        // ==========================================

        // --- Component 0 (x-direction) ---
        //s_wsp0(erqp) . s_basis(ip) = s_wsp1(erqi)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int q = row % nq;

                const int p = col;
            
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };
        
            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                const int p = row;
                const int i = col;

                return s_basis[i * nq + p];
            };
        
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nq);
                const int r = (row / nq) % nq;
                const int q = row % nq;

                const int i = col;
            
                return s_wsp1[e * (nq * nq * nm) + r * (nq * nm) + q * nm + i];
            };
        
            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm, nq>(v_s_wsp0, v_s_basis, v_s_wsp1);
        }


        //s_wsp1(erqi) . s_basis(jq) = s_wsp0(erij)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nm);
                const int r = (row / nm) % nq;
                const int i = row % nm;

                const int q = col;

                return s_wsp1[e * (nq * nq * nm) + r * (nq * nm) + q * nm + i];
            };

            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                const int q = row;
                const int j = col;

                return s_basis[j * nq + q];
            };

            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nq * nm);
                const int r = (row / nm) % nq;
                const int i = row % nm;

                const int j = col;

                return s_wsp0[e * (nq * nm * nm) + r * (nm * nm) + i * nm + j];
            };

            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nm, nm, nq>(v_s_wsp1, v_s_basis, v_s_wsp0);
        }

        //s_wsp0(erij) . s_basis(kr) = s_wsp1(eijk)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nm * nm);
                const int i = (row / nm) % nm;
                const int j = row % nm;

                const int r = col;
                    
                return s_wsp0[e * (nq * nm * nm) + r * (nm * nm) + i * nm + j];
            };

            auto v_s_basis = [=] __device__  (const int row, const int col) -> double& {
                const int r = row;
                const int k = col;
            
                return s_basis[k * nq + r];
            };
        
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int e = row / (nm * nm);
                const int i = (row / nm) % nm;
                const int j = row % nm;

                const int k = col;
            
                return s_wsp1[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];
            };
        
            f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nm, nm, nq>(v_s_wsp0, v_s_basis, v_s_wsp1);
        }




        // ==========================================
        // PHASE 5: Write to Output
        // ==========================================
        for(int tid = threadIdx.x; tid < nelmtPerBatch * ndof_1D; tid += blockDim.x) {
    
            int e = tid / ndof_1D;
            int dof = tid % ndof_1D;

            d_out[comp_offset + batch_offset + e * nm*nm*nm + dof] = s_wsp1[tid];
        }
        __syncwarp();
        
        } //component loop
        eb += gridDim.x;
    }   
}




} //namespace Parallel
#endif //MMA_F64_M8N8K4_MASS_OPERATOR_CUH