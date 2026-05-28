#ifndef MMA_F64_M8N8K4_MASS_OPERATOR_CUH
#define MMA_F64_M8N8K4_MASS_OPERATOR_CUH


namespace Parallel {

enum class Layout{RowMajor, ColMajor};

template<typename T, Layout L, size_t rows, size_t cols>
__device__ auto matrixView(T* data) {
    return [=](const size_t i, const size_t j) -> T& {
        if (L == Layout::RowMajor) {
            return data[i * cols + j];
        } else {
            return data[j * rows + i];
        }
    };
}


template<int M, int N, int K, Layout Layout_A, Layout Layout_B, Layout Layout_C>
__device__ void f64_m8n8k4_tiled_gemm(double *s_A, double *s_B, double *s_C)
{   
    constexpr int m = 8;
    constexpr int n = 8;
    constexpr int k = 4;

    const int tid = threadIdx.x;
    const int laneid = tid % warpSize;

    constexpr int num_tiles_m = (M + m - 1) / m;
    constexpr int num_tiles_n = (N + n - 1) / n;
    constexpr int num_tiles_k = (K + k - 1) / k;

    double r_b[num_tiles_k][num_tiles_n] = {0};

    auto s_B_view = matrixView<double, Layout_B, K, N>(s_B);

    //copy s_B from shared memory to register
    {
    int base_row = laneid % 4;
    int base_col = laneid >> 2;
    
    int row = base_row;
    int col = base_col;
    
    #pragma unroll
    for(int i = 0; i < num_tiles_k; i++){
        row = base_row + i * k;

        #pragma unroll
        for(int j = 0; j < num_tiles_n; j++){
            col = base_col + j * n;
            if(row < K && col < N){
                r_b[i][j] = s_B_view(row, col);
            } 
            else{
                r_b[i][j] = 0.0;
            }
        }
    }
    __syncwarp();
    }


    auto s_A_view = matrixView<double, Layout_A, M, K>(s_A);

    double r_a[num_tiles_m][num_tiles_k] = {0};
    double r_c[num_tiles_m][num_tiles_n][2] = {0};
        
    //copy s_A from shared memory to register
    {
        const int base_row = laneid >> 2;
        const int base_col = laneid % 4;

        int row = base_row;
        int col = base_col;

        #pragma unroll
        for(int i = 0; i < num_tiles_m; i++){
            row = base_row + i * m;
    
            #pragma unroll
            for(int j = 0; j < num_tiles_k; j++){
                col = base_col + j * k;
                if(row < M && col < K){
                    r_a[i][j] = s_A_view(row, col);
                }
                else{
                    r_a[i][j] = 0.0;
                }
            }
        }
        __syncwarp();
    }

    //tiled GEMM
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
                    :"+d"(r_c[i][j][0]), "+d"(r_c[i][j][1])
                    :"d"(r_a[i][t]),
                     "d"(r_b[t][j])
                );
            }
        }
    }

    auto s_C_view = matrixView<double, Layout_C, M, N>(s_C);

    {
        //copy from register to shared memory s_C
        const int base_row = laneid >> 2;
        const int base_col = (laneid % 4) * 2;
        
        int row = base_row;
        int col = base_col;
        #pragma unroll
        for(int i = 0; i < num_tiles_m; ++i){
            row = base_row + i * m;
            #pragma unroll
            for(int j = 0; j < num_tiles_n; ++j){
                col = base_col + j * n;
                if(row < M && col < N){
                    s_C_view(row, col) = r_c[i][j][0];
                }
                col += 1;
                if(row < M && col < N){
                    s_C_view(row, col) = r_c[i][j][1];
                }
            }
        }
        __syncwarp();
    }
}


template<const unsigned int nq, const unsigned int nm_t, const unsigned int nm_n, const unsigned int nelmtPerBatch>
void __global__ f64_m8n8k4_Mass_mma(
    const unsigned int nelmt,
    const unsigned int numBlocks, const unsigned int threadsPerBlock,
    const double *__restrict__ d_basis_n, const double *__restrict__ d_basis_t,
    const double *__restrict__ d_G, double *__restrict__ d_in, double * __restrict__ d_out, const unsigned int ntests) 
{
    using T = double;

    // Total DoFs per element for a Raviart-Thomas vector
    constexpr unsigned int ndof_1D = nm_n * nm_t * nm_t;
    constexpr unsigned int ndof_total = ndof_1D + ndof_1D + ndof_1D;

    extern __shared__ T scratch[];
    T *s_basis_n  = scratch;
    T *s_basis_t  = s_basis_n + nm_n * nq;

    T *s_wsp0    = s_basis_t + nm_t * nq;
    T *s_wsp1    = s_wsp0    + nelmtPerBatch * nq * nq * nq;

    T *s_uq_0     = s_wsp1   + nelmtPerBatch * nq * nq * nq;
    T *s_uq_1     = s_uq_0   + nelmtPerBatch * nq * nq * nq;
    T *s_uq_2     = s_uq_1   + nelmtPerBatch * nq * nq * nq;;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nm_t * nq; tid += blockDim.x)
    {
        s_basis_t[tid] = d_basis_t[tid];
    }


    for(unsigned int tid = threadIdx.x; tid < nm_n * nq; tid += blockDim.x)
    {
        s_basis_n[tid] = d_basis_n[tid];
    }
    __syncwarp();



    //element batch iteration
    unsigned int eb = blockIdx.x;

    while(eb < (nelmt + nelmtPerBatch - 1) / nelmtPerBatch)
    {
        const int global_batch_offset = eb * nelmtPerBatch * ndof_total;

        //step-1 : Copy from in to the uq
        for(int tid = threadIdx.x; tid < nelmtPerBatch * ndof_1D; tid += blockDim.x) {
            int e = tid / ndof_1D;
            int dof = tid % ndof_1D;

            s_uq_0[tid] = d_in[global_batch_offset + e * ndof_total + 0 * ndof_1D + dof];
            s_uq_1[tid] = d_in[global_batch_offset + e * ndof_total + 1 * ndof_1D + dof];
            s_uq_2[tid] = d_in[global_batch_offset + e * ndof_total + 2 * ndof_1D + dof];
        }
        __syncwarp();

        // ==========================================
        // PHASE 1: Interpolate to Quadrature Nodes
        // ==========================================

        // --- Component 0 (x-direction) ---

        //s_uq0(eijk) . s_basis_t(kr) = s_wsp0(eijr)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_n * nm_t, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_uq_0, s_basis_t, s_wsp0);

        //s_wsp0(eijr) -> s_wsp1(eirj)
        for(int tid = threadIdx.x; tid < nq * nm_t; tid += blockDim.x){
            int j = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_n; ++i) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_n) + i*nq*nm_t + j*nq + r];

                    s_wsp1[e * (nq*nm_t*nm_n) + i*nq*nm_t + r*nm_t + j] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eirj) . s_basis_t(jq) = s_wsp0(eirq)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_n * nq, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_wsp0);

        //s_wsp0(eirq) -> s_wsp1(erqi)
        for(int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x){
            int r = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_n; ++i) {
                    T reg = s_wsp0[e * (nq*nq*nm_n) + i*nq*nq + r*nq + q];
                    s_wsp1[e * (nq*nq*nm_n) + r*nq*nm_n + q*nm_n + i] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(erqi) . s_basis_n(ip) = s_uq_0(erqp)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nm_n, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_n, s_uq_0);

        



        // --- Component 1 (y-direction) ---

        //s_uq1(eijk) . s_basis_t(kr) = s_wsp0(eijr)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_n * nm_t, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_uq_1, s_basis_t, s_wsp0);

        //s_wsp0(eijr) -> s_wsp1(eirj)
        for(int tid = threadIdx.x; tid < nq * nm_n; tid += blockDim.x){
            int j = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_t; ++i) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_n) + i*nq*nm_n + j*nq + r];

                    s_wsp1[e * (nq*nm_t*nm_n) + i*nq*nm_n + r*nm_n + j] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eirj) . s_basis_t(jq) = s_wsp0(eirq)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nq, nq, nm_n, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_n, s_wsp0);

        //s_wsp0(eirq) -> s_wsp1(erqi)
        for(int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x){
            int r = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_t; ++i) {
                    T reg = s_wsp0[e * (nq*nq*nm_t) + i*nq*nq + r*nq + q];

                    s_wsp1[e * (nq*nq*nm_t) + r*nq*nm_t + q*nm_t + i] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(erqi) . s_basis_n(ip) = s_uq_1(erqp)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_uq_1);






        // --- Component 2 (z-direction) ---

        //s_uq1(eijk) . s_basis_n(kr) = s_wsp0(eijr)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nm_t, nq, nm_n, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_uq_2, s_basis_n, s_wsp0);

        //s_wsp0(eijr) -> s_wsp1(eirj)
        for(int tid = threadIdx.x; tid < nq * nm_t; tid += blockDim.x){
            int j = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_t; ++i) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_t) + i*nq*nm_t + j*nq + r];

                    s_wsp1[e * (nq*nm_t*nm_t) + i*nq*nm_t + r*nm_t + j] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eirj) . s_basis_t(jq) = s_wsp0(eirq)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nq, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_wsp0);

        //s_wsp0(eirq) -> s_wsp1(erqi)
        for(int tid = threadIdx.x; tid < nq * nq; tid += blockDim.x){
            int r = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_t; ++i) {
                    T reg = s_wsp0[e * (nq*nq*nm_t) + i*nq*nq + r*nq + q];

                    s_wsp1[e * (nq*nq*nm_t) + r*nq*nm_t + q*nm_t + i] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(erqi) . s_basis_n(ip) = s_uq_1(erqp)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nm_t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_uq_2);



        // ==========================================
        // PHASE 2: Apply Piola Geometry Metric
        // ==========================================

        for(unsigned int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){

            int e = tid / (nq * nq);
            int q = (tid / nq) % nq; 
            int p = tid % nq;  

            int e_offset = eb * nelmtPerBatch * 6 * nq*nq*nq + e * 6 * nq*nq*nq;

            T G00, G01, G02, G11, G12, G22;
            T u0, u1, u2;

            for(unsigned int r = 0; r < nq; ++r){

                int G_idx = e_offset + r * nq*nq + q * nq + p;

                G00 = d_G[0 * nq*nq*nq + G_idx];
                G01 = d_G[1 * nq*nq*nq + G_idx];
                G02 = d_G[2 * nq*nq*nq + G_idx];
                G11 = d_G[3 * nq*nq*nq + G_idx];
                G12 = d_G[4 * nq*nq*nq + G_idx];
                G22 = d_G[5 * nq*nq*nq + G_idx];
                    
                int shm_idx = e * nq*nq*nq + r * nq*nq + q * nq + p;
                    
                u0 = s_uq_0[shm_idx];
                u1 = s_uq_1[shm_idx];
                u2 = s_uq_2[shm_idx];
                    
                s_uq_0[shm_idx] = G00 * u0 + G01 * u1 + G02 * u2;
                s_uq_1[shm_idx] = G01 * u0 + G11 * u1 + G12 * u2;
                s_uq_2[shm_idx] = G02 * u0 + G12 * u1 + G22 * u2;
            }
        }
        __syncwarp();


        // ==========================================
        // PHASE 3: Project back to Nodes
        // ==========================================

        // --- Component 0 (x-direction) ---

        //s_uq_0(erqp) . s_basis_n(pi) = s_wsp0(erqi)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm_n, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_uq_0, s_basis_n, s_wsp0);

        //s_wsp0(erqi) -> s_wsp1(eriq)
        for(int tid = threadIdx.x; tid < nm_n * nq; tid += blockDim.x){
            int i = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int r = 0; r < nq; ++r) {
                    T reg = s_wsp0[e * (nq*nq*nm_n) + r*nq*nm_n + q*nm_n + i];

                    s_wsp1[e * (nq*nq*nm_n) + r*nq*nm_n + i*nq + q] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eriq) . s_basis_t(qj) = s_wsp0(erij)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nm_n, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_wsp0);

        //s_wsp0(erij) -> s_wsp1(eijr)
        for(int tid = threadIdx.x; tid < nq * nm_t; tid += blockDim.x){
            int j = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_n; ++i) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_n) + r*nm_t*nm_n + i*nm_t + j];

                    s_wsp1[e * (nq*nm_t*nm_n) + i*nq*nm_t + j*nq + r] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eijr) . s_basis_t(rk) = s_uq_0(eijk)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nm_n, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_uq_0);







        // --- Component 1 (y-direction) ---

        //s_uq1(erqp) . s_basis_t(pi) = s_wsp0(erqi)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_uq_1, s_basis_t, s_wsp0);

        //s_wsp0(erqi) -> s_wsp1(eriq)
        for(int tid = threadIdx.x; tid < nm_t * nq; tid += blockDim.x){
            int i = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int r = 0; r < nq; ++r) {
                    T reg = s_wsp0[e * (nq*nq*nm_t) + r*nq*nm_t + q*nm_t + i];

                    s_wsp1[e * (nq*nq*nm_t) + r*nq*nm_t + i*nq + q] = reg;
                }
            }
        }
        __syncwarp();


        //s_wsp1(eriq) . s_basis_n(qj) = s_wsp0(erij)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nm_t, nm_n, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_n, s_wsp0);

        //s_wsp0(erij) -> s_wsp1(eijr)
        for(int tid = threadIdx.x; tid < nq * nm_n; tid += blockDim.x){
            int j = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int i = 0; i < nm_t; ++i) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_n) + r*nm_t*nm_n + i*nm_n + j];

                    s_wsp1[e * (nq*nm_t*nm_n) + i*nq*nm_n + j*nq + r] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eijr) . s_basis_t(rk) = s_uq_1(eijk)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nm_n, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_uq_1);




        // --- Component 2 (z-direction) ---

        //s_uq2(erqp) . s_basis_t(pi) = s_wsp0(erqi)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_uq_2, s_basis_t, s_wsp0);

        //s_wsp0(erqi) -> s_wsp1(eriq)
        for(int tid = threadIdx.x; tid < nm_t * nq; tid += blockDim.x){
            int i = tid / nq;
            int q = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int r = 0; r < nq; ++r) {
                    T reg = s_wsp0[e * (nq*nq*nm_t) + r*nq*nm_t + q*nm_t + i];

                    s_wsp1[e * (nq*nq*nm_t) + r*nq*nm_t + i*nq + q] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eriq) . s_basis_n(qj) = s_wsp0(erij)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nm_t, nm_t, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_t, s_wsp0);

        //s_wsp0(erij) -> s_wsp1(eijr)
        for(int tid = threadIdx.x; tid < nq * nm_t; tid += blockDim.x){
            int i = tid / nq;
            int r = tid % nq;

            for (int e = 0; e < nelmtPerBatch; ++e) {
                for (int j = 0; j < nm_t; ++j) {
                    T reg = s_wsp0[e * (nq*nm_t*nm_t) + r*nm_t*nm_t + i*nm_t + j];

                    s_wsp1[e * (nq*nm_t*nm_t) + i*nq*nm_t + j*nq + r] = reg;
                }
            }
        }
        __syncwarp();

        //s_wsp1(eijr) . s_basis_t(rk) = s_uq_2(eijk)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm_t * nm_t, nm_n, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_basis_n, s_uq_2);




        // ==========================================
        // PHASE 4: Write to Output
        // ==========================================
        const int global_batch_offset_out = eb * nelmtPerBatch * ndof_total;

        for(int tid = threadIdx.x; tid < nelmtPerBatch * ndof_1D; tid += blockDim.x) {
    
            int e = tid / ndof_1D;
            int dof = tid % ndof_1D;

            d_out[global_batch_offset_out + e * ndof_total + 0 * ndof_1D + dof] = s_uq_0[tid];
            d_out[global_batch_offset_out + e * ndof_total + 1 * ndof_1D + dof] = s_uq_1[tid];
            d_out[global_batch_offset_out + e * ndof_total + 2 * ndof_1D + dof] = s_uq_2[tid];
        }
        __syncwarp();
    
        eb += gridDim.x;
    }   
}




} //namespace Parallel
#endif //MMA_F64_M8N8K4_MASS_OPERATOR_CUH