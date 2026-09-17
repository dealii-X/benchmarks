#ifndef MMA_F64_M8N8K4_MASS_CUH
#define MMA_F64_M8N8K4_MASS_CUH

namespace Parallel {

template<int num_tiles_k, int num_tiles_n>
struct RegMatrixB {
    double r_b[num_tiles_k][num_tiles_n];
};

template<int N, int K, typename FetchB>
__device__ RegMatrixB<(K + 3) / 4, (N + 7) / 8> load_matrix_B_to_regs(FetchB get_B) {
    constexpr int k = 4;
    constexpr int n = 8;
    constexpr int num_tiles_k = (K + k - 1) / k;
    constexpr int num_tiles_n = (N + n - 1) / n;
    
    RegMatrixB<num_tiles_k, num_tiles_n> B_regs;

    const int laneid = threadIdx.x % warpSize;
    int base_row = laneid % 4;
    int base_col = laneid >> 2;

    #pragma unroll
    for(int i = 0; i < num_tiles_k; i++){
        int row = base_row + i * k;
        #pragma unroll
        for(int j = 0; j < num_tiles_n; j++){
            int col = base_col + j * n;
            if(row < K && col < N) {
                B_regs.r_b[i][j] = get_B(row, col);
            } else {
                B_regs.r_b[i][j] = 0.0;
            }
        }
    }
    return B_regs;
}

template<int M, int N, int K, bool Accumulate = false, typename FetchA, typename StoreC>
__device__ void f64_m8n8k4_tiled_gemm(
    FetchA get_A, 
    const RegMatrixB<(K + 3) / 4, (N + 7) / 8>& B_regs, 
    StoreC set_C, 
    const int valid_M = M)
{   
    constexpr int m = 8;
    constexpr int n = 8;
    constexpr int k = 4;

    const int laneid = threadIdx.x % warpSize;

    constexpr int num_tiles_m = (M + m - 1) / m;
    constexpr int num_tiles_n = (N + n - 1) / n;
    constexpr int num_tiles_k = (K + k - 1) / k;

    double r_a[num_tiles_m][num_tiles_k] = {0.0};
    double r_c[num_tiles_m][num_tiles_n][2] = {0.0};
        
    // 1. Copy Matrix A from shared memory to registers
    {
        const int base_row = laneid >> 2;
        const int base_col = laneid % 4;

        #pragma unroll
        for(int i = 0; i < num_tiles_m; i++){
            int row = base_row + i * m;
            #pragma unroll
            for(int j = 0; j < num_tiles_k; j++){
                int col = base_col + j * k;
                // Mask out-of-bounds M accesses!
                if(row < valid_M && col < K){
                    r_a[i][j] = get_A(row, col);
                }
                else{
                    r_a[i][j] = 0.0;
                }
            }
        }
    }

    // 2. Tiled Tensor Core MMA Computation
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
                      "d"(B_regs.r_b[t][j])
                );
            }
        }
    }

    // 3. Copy accumulator results to destination memory
    {
        const int base_row = laneid >> 2;
        const int base_col = (laneid % 4) * 2;
        
        #pragma unroll
        for(int i = 0; i < num_tiles_m; ++i){
            int row = base_row + i * m;
            #pragma unroll
            for(int j = 0; j < num_tiles_n; ++j){
                int col = base_col + j * n;
                
                if(row < valid_M && col < N){
                    if constexpr (Accumulate) set_C(row, col) += r_c[i][j][0];
                    else                      set_C(row, col)  = r_c[i][j][0];
                }
                
                int col_next = col + 1;
                if(row < valid_M && col_next < N){
                    if constexpr (Accumulate) set_C(row, col_next) += r_c[i][j][1];
                    else                      set_C(row, col_next)  = r_c[i][j][1];
                }
            }
        }
    }
}


// =========================================================================
// MAIN KERNEL
// =========================================================================
template<const unsigned int nq, const unsigned int nm, const unsigned int nelmtPerBatch>
void __global__ f64_m8n8k4_mma(
    const unsigned int nelmt,
    const double *__restrict__ d_basis, const double *__restrict__ d_dbasis,
    const double *__restrict__ d_G, double *__restrict__ d_in, double * __restrict__ d_out, const unsigned int ntests) 
{
    using T = double;
    constexpr unsigned int ndof_1D = nm * nm * nm;

    extern __shared__ T scratch[];
    T *s_basis  = scratch;
    T *s_dbasis = s_basis  + nq * nm;

    T *s_wsp0   = s_dbasis + nq * nq;
    T *s_wsp1   = s_wsp0   + nelmtPerBatch * nq * nq * nq;

    T *s_rqr    = s_wsp1   + nelmtPerBatch * nq * nq * nq;
    T *s_rqs    = s_rqr    + nelmtPerBatch * nq * nq * nq;
    T *s_rqt    = s_wsp1;

    const int warpid    = threadIdx.x / warpSize;
    const int num_warps = blockDim.x / warpSize;

    // copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nm * nq; tid += blockDim.x) {
        s_basis[tid] = d_basis[tid];
    }
    for(unsigned int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x) {
        s_dbasis[tid] = d_dbasis[tid];
    }
    __syncthreads();


    // Element batch iteration
    unsigned int eb = blockIdx.x;

    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
    {
        for(int c = 0; c < 3; ++c)
        {
        const size_t comp_offset  = c * nelmt * nm * nm * nm;
        const size_t batch_offset = eb * nelmtPerBatch * nm * nm * nm;

        // step-1 : Copy from in to the wsp0
        for(int i = threadIdx.x; i < nelmtPerBatch * nm * nm * nm; i += blockDim.x) {
            s_wsp1[i] = d_in[comp_offset + batch_offset + i];
        }
        __syncthreads();


        // ==========================================
        // PHASE 1: Interpolate to Quadrature Nodes
        // ==========================================
        
        // HOIST: Load s_basis into registers for ALL Phase 1 loops
        auto v_s_basis_P1 = [=] __device__ (const int row, const int col) -> double& {
            return s_basis[row * nq + col];
        };
        auto B_basis_P1 = load_matrix_B_to_regs<nq, nm>(v_s_basis_P1);

        const int M1_0 = nelmtPerBatch * nm * nm;
        for(int m_offset = warpid * 8; m_offset < M1_0; m_offset += num_warps * 8)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e   = r_idx / (nm * nm);
                const int i   = (r_idx / nm) % nm;
                const int j   = r_idx % nm;
                const int k   = col;
                return s_wsp1[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];
            };

            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nm * nm);
                const int i = (r_idx / nm) % nm;
                const int j = r_idx % nm;
                const int r = col;
                return s_wsp0[e * (nm * nm * nq) + i * (nm * nq) + j * nq + r];
            };

            f64_m8n8k4_tiled_gemm<8, nq, nm>(v_s_wsp1, B_basis_P1, v_s_wsp0, M1_0 - m_offset);
        }
        __syncthreads();

        const int M1_1 = nelmtPerBatch * nm * nq;
        for(int m_offset = warpid * 8; m_offset < M1_1; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nm * nq);
                const int i = (r_idx / nq) % nm;
                const int r = r_idx % nq;
                const int j = col;
                return s_wsp0[e * (nm * nm * nq) + i * (nm * nq) + j * nq + r];
            };

            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nm * nq);
                const int i = (r_idx / nq) % nm;
                const int r = r_idx % nq;
                const int q = col;
                return s_wsp1[e * (nm * nq * nq) + i * (nq * nq) + r * nq + q];
            };

            f64_m8n8k4_tiled_gemm<8, nq, nm>(v_s_wsp0, B_basis_P1, v_s_wsp1, M1_1 - m_offset);
        }
        __syncthreads();

        const int M1_2 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M1_2; m_offset += num_warps * 8)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int i = col;
                return s_wsp1[e * (nm * nq * nq) + i * (nq * nq) + r * nq + q];
            };
        
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int p = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };
        
            f64_m8n8k4_tiled_gemm<8, nq, nm>(v_s_wsp1, B_basis_P1, v_s_wsp0, M1_2 - m_offset);
        }
        __syncthreads();


        // ==========================================
        // PHASE 2: Apply Grad on Quad. Pts.
        // ==========================================
        
        // HOIST: Load s_dbasis into registers for ALL Phase 2 loops
        auto v_s_dbasis_P2 = [=] __device__ (const int row, const int col) -> double& {
            return s_dbasis[row * nq + col];
        };
        auto B_dbasis_P2 = load_matrix_B_to_regs<nq, nq>(v_s_dbasis_P2);

        const int M2_0 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M2_0; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int p = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };

            auto v_s_rqr = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq; 
                const int q = r_idx % nq;        
                const int i = col;             
                return s_rqr[e * (nq * nq * nq) + i * (nq * nq) + q * nq + r];
            };
        
            f64_m8n8k4_tiled_gemm<8, nq, nq>(v_s_wsp0, B_dbasis_P2, v_s_rqr, M2_0 - m_offset);
        }
        __syncthreads();
        
        const int M2_1 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M2_1; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int p = r_idx % nq;
                const int q = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };

            auto v_s_rqs = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq; 
                const int p = r_idx % nq;        
                const int j = col;             
                return s_rqs[e * (nq * nq * nq) + p * (nq * nq) + j * nq + r];
            };

            f64_m8n8k4_tiled_gemm<8, nq, nq>(v_s_wsp0, B_dbasis_P2, v_s_rqs, M2_1 - m_offset);
        }
        __syncthreads();

        const int M2_2 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M2_2; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int r = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };

            auto v_s_rqt = [=] __device__ (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq; 
                const int q = r_idx % nq;        
                const int k = col;             
                return s_rqt[e * (nq * nq * nq) + p * (nq * nq) + q * nq + k];
            };

            f64_m8n8k4_tiled_gemm<8, nq, nq>(v_s_wsp0, B_dbasis_P2, v_s_rqt, M2_2 - m_offset);
        }
        __syncthreads();
        

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
        __syncthreads(); 


        // ==========================================
        // PHASE 4: Apply Divergence
        // ==========================================
        
        // HOIST: Load transposed s_dbasis into registers for ALL Phase 4 loops
        auto v_s_dbasis_T_P4 = [=] __device__ (const int row, const int col) -> double& {
            return s_dbasis[col * nq + row]; 
        };
        auto B_dbasis_T_P4 = load_matrix_B_to_regs<nq, nq>(v_s_dbasis_T_P4);

        const int M4_0 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M4_0; m_offset += num_warps * 8)
        {
            auto v_s_rqr = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int q = (r_idx / nq) % nq;
                const int r = r_idx % nq;
                const int n = col;
                return s_rqr[e * (nq*nq*nq) + n * (nq*nq) + q * nq + r];
            };

            auto v_s_wsp0 = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int q = (r_idx / nq) % nq;
                const int r = r_idx % nq;
                const int p = col;
                return s_wsp0[e * (nq*nq*nq) + r * (nq*nq) + q * nq + p];
            };

            f64_m8n8k4_tiled_gemm<8, nq, nq>(v_s_rqr, B_dbasis_T_P4, v_s_wsp0, M4_0 - m_offset);
        }
        __syncthreads();
            
        const int M4_1 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M4_1; m_offset += num_warps * 8)
        {
            auto v_s_rqs = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq;
                const int r = r_idx % nq;
                const int n = col;
                return s_rqs[e * (nq*nq*nq) + p * (nq*nq) + n * nq + r];
            };

            auto v_s_wsp0 = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq;
                const int r = r_idx % nq;
                const int q = col;
                return s_wsp0[e * (nq*nq*nq) + r * (nq*nq) + q * nq + p];
            };

            // Notice: Accumulate = true
            f64_m8n8k4_tiled_gemm<8, nq, nq, true>(v_s_rqs, B_dbasis_T_P4, v_s_wsp0, M4_1 - m_offset);
        }
        __syncthreads();
        
        const int M4_2 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M4_2; m_offset += num_warps * 8)
        {
            auto v_s_rqt = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int n = col;
                return s_rqt[e * (nq*nq*nq) + p * (nq*nq) + q * nq + n];
            };

            auto v_s_wsp0 = [=] __device__ (int row, int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int p = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int r = col;
                return s_wsp0[e * (nq*nq*nq) + r * (nq*nq) + q * nq + p];
            };
            
            // Notice: Accumulate = true
            f64_m8n8k4_tiled_gemm<8, nq, nq, true>(v_s_rqt, B_dbasis_T_P4, v_s_wsp0, M4_2 - m_offset);
        }
        __syncthreads();


        // ==========================================
        // PHASE 5: Project back to Nodes
        // ==========================================

        // HOIST: Load transposed s_basis into registers for ALL Phase 5 loops
        auto v_s_basis_T_P5 = [=] __device__ (const int row, const int col) -> double& {
            return s_basis[col * nq + row]; 
        };
        auto B_basis_T_P5 = load_matrix_B_to_regs<nm, nq>(v_s_basis_T_P5);

        const int M5_0 = nelmtPerBatch * nq * nq;
        for(int m_offset = warpid * 8; m_offset < M5_0; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int p = col;
                return s_wsp0[e * (nq * nq * nq) + r * (nq * nq) + q * nq + p];
            };
        
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nq);
                const int r = (r_idx / nq) % nq;
                const int q = r_idx % nq;
                const int i = col;
                return s_wsp1[e * (nq * nq * nm) + r * (nq * nm) + q * nm + i];
            };
        
            f64_m8n8k4_tiled_gemm<8, nm, nq>(v_s_wsp0, B_basis_T_P5, v_s_wsp1, M5_0 - m_offset);
        }
        __syncthreads();

        const int M5_1 = nelmtPerBatch * nq * nm;
        for(int m_offset = warpid * 8; m_offset < M5_1; m_offset += num_warps * 8)
        {
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nm);
                const int r = (r_idx / nm) % nq;
                const int i = r_idx % nm;
                const int q = col;
                return s_wsp1[e * (nq * nq * nm) + r * (nq * nm) + q * nm + i];
            };

            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nq * nm);
                const int r = (r_idx / nm) % nq;
                const int i = r_idx % nm;
                const int j = col;
                return s_wsp0[e * (nq * nm * nm) + r * (nm * nm) + i * nm + j];
            };

            f64_m8n8k4_tiled_gemm<8, nm, nq>(v_s_wsp1, B_basis_T_P5, v_s_wsp0, M5_1 - m_offset);
        }
        __syncthreads();

        const int M5_2 = nelmtPerBatch * nm * nm;
        for(int m_offset = warpid * 8; m_offset < M5_2; m_offset += num_warps * 8)
        {
            auto v_s_wsp0 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nm * nm);
                const int i = (r_idx / nm) % nm;
                const int j = r_idx % nm;
                const int r = col;
                return s_wsp0[e * (nq * nm * nm) + r * (nm * nm) + i * nm + j];
            };
        
            auto v_s_wsp1 = [=] __device__  (const int row, const int col) -> double& {
                const int r_idx = m_offset + row;
                const int e = r_idx / (nm * nm);
                const int i = (r_idx / nm) % nm;
                const int j = r_idx % nm;
                const int k = col;
                return s_wsp1[e * (nm * nm * nm) + i * (nm * nm) + j * nm + k];
            };
        
            f64_m8n8k4_tiled_gemm<8, nm, nq>(v_s_wsp0, B_basis_T_P5, v_s_wsp1, M5_2 - m_offset);
        }
        __syncthreads();


        // ==========================================
        // PHASE 6: Write to Output
        // ==========================================
        for(int tid = threadIdx.x; tid < nelmtPerBatch * ndof_1D; tid += blockDim.x) {
            int e = tid / ndof_1D;
            int dof = tid % ndof_1D;
            d_out[comp_offset + batch_offset + e * nm*nm*nm + dof] = s_wsp1[tid];
        }
        __syncthreads();
        
        } // component loop
        eb += gridDim.x;
    }   
}

} // namespace Parallel
#endif // MMA_F64_M8N8K4_MASS_CUH