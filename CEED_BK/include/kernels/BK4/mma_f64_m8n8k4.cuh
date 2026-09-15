#ifndef MMA_F64_M8N8K4_MASS_CUH
#define MMA_F64_M8N8K4_MASS_CUH


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
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nm, nq, nm, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis, s_wsp0);

        //s_wsp0(eijr) -> s_wsp1(eirj)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nm * nq; tid += blockDim.x){
            int e = tid / (nm * nq);
            int j = (tid / nq) % nm;
            int r = tid % nq;
        
            double r_tmp[nm];

            for (int i = 0; i < nm; ++i) {
                r_tmp[i] = s_wsp0[e * (nq*nm*nm) + i*nq*nm + j*nq + r];
            }

            for (int i = 0; i < nm; ++i) {
                s_wsp1[e * (nq*nm*nm) + i*nq*nm + r*nm + j] = r_tmp[i];
            }
        }
        __syncwarp();

        //s_wsp1(eirj) . s_basis(jq) = s_wsp0(eirq)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nq, nq, nm, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis, s_wsp0);

        //s_wsp0(eirq) -> s_wsp1(erqi)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
            int e = tid / (nq * nq);
            int r = (tid / nq) % nq;
            int q = tid % nq;
        
            double r_tmp[nm];

            for (int i = 0; i < nm; ++i) {
                r_tmp[i] = s_wsp0[e * (nq*nq*nm) + i*nq*nq + r*nq + q];
            }

            for (int i = 0; i < nm; ++i) {
                s_wsp1[e * (nq*nq*nm) + r*nq*nm + q*nm + i] = r_tmp[i];
            }
        }
        __syncwarp();

        //s_wsp1(erqi) . s_basis(ip) = s_wsp0(erqp)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nq, nm, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor>(s_wsp1, s_basis, s_wsp0);


        // ==========================================
        // PHASE 2: Apply Grad on Quad. Pts.
        // ==========================================
        /*
        //s_wsp0(erqp) . s_dbasis(ip) = s_rqr(erqi)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_dbasis, s_rqr);

        //s_wsp0(erqp) -> s_wsp1(erpq)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
            int e = tid / (nq * nq);
            int r = (tid / nq) % nq;
            int q = tid % nq;
        
            double r_tmp[nq];
        
            for (int p = 0; p < nq; ++p) {
                r_tmp[p] = s_wsp0[e * (nq*nq*nq) + r * (nq*nq) + q * nq + p];
            }
        
            for (int p = 0; p < nq; ++p) {
                s_wsp1[e * (nq*nq*nq) + r * (nq*nq) + p * nq + q] = r_tmp[p];
            }
        }
        __syncwarp();

        //s_wsp1(erpq) . s_dbasis(jq) = s_rqs(erpj)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_dbasis, s_rqs);

        //s_wsp0(erpq) -> s_wsp1(epqr)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
            int e = tid / (nq * nq);
            int p = (tid / nq) % nq;
            int q = tid % nq;

            double r_tmp[nq];

            for (int r = 0; r < nq; ++r) {
                r_tmp[r] = s_wsp0[e * (nq*nq*nq) + r * (nq*nq) + p * nq + q];
            }

            for (int r = 0; r < nq; ++r) {
                s_wsp1[e * (nq*nq*nq) + p * (nq*nq) + q * nq + r] = r_tmp[r];
            }
        }
        __syncwarp();

        //s_wsp1(epqr) . s_dbasis(kr) = s_rqt(epqk)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp1, s_dbasis, s_rqt);
        */

        // ==========================================
        // PHASE 3: Apply G
        // ==========================================
        double r_p[nq], r_q[nq], r_r[nq];

        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){

            int e = tid / (nq * nq);
            int q = tid % (nq * nq) / nq;
            int r = tid % nq;


            //copy to register
            for(int n = 0; n < nq; n++)
            {
                r_p[n] = s_wsp0[e * nq*nq*nq + r * nq*nq + q * nq + n];
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
                    qs += r_q[n] * s_wsp0[e * nq*nq*nq + r * nq*nq + n * nq + p];
                    qt += r_r[n] * s_wsp0[e * nq*nq*nq + n * nq*nq + q * nq + p];
                }
            
                // Apply chain rule
                s_rqr[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grr * qt + Grs * qs + Grt * qr;
                s_rqs[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grs * qt + Gss * qs + Gst * qr;
                s_rqt[e * nq*nq*nq + p * nq * nq + q * nq + r] = Grt * qt + Gst * qs + Gtt * qr;
            }
        }
        __syncwarp();


                        
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
        
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

                s_wsp0[e * nq*nq*nq + r * nq*nq + q * nq + p] = tmp0;
            }
        }
        __syncwarp();

        
        // ==========================================
        // PHASE 4: Project back to Nodes
        // ==========================================

        // --- Component 0 (x-direction) ---
        //s_wsp0(erqp) . s_basis(ip) = s_wsp1(erqi)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nq, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp0, s_basis, s_wsp1);

        //s_wsp1(erqi) -> s_wsp0(eriq)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nq * nq; tid += blockDim.x){
            int e = tid / (nq * nq);
            int r = (tid / nq) % nq;
            int q = tid % nq;
        
            double r_tmp[nm];
        
            for (int i = 0; i < nm; ++i) {
                r_tmp[i] = s_wsp1[e * (nq*nq*nm) + r * (nq*nm) + q * nm + i];
            }
        
            for (int i = 0; i < nm; ++i) {
                s_wsp0[e * (nq*nq*nm) + r * (nq*nm) + i * nq + q] = r_tmp[i];
            }
        }
        __syncwarp();

        //s_wsp0(eriq) . s_basis(jq) = s_wsp1(erij)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nq * nm, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp0, s_basis, s_wsp1);

        //s_wsp1(erij) -> s_wsp0(eijr)
        for(int tid = threadIdx.x; tid < nelmtPerBatch * nm * nm; tid += blockDim.x){
            int e = tid / (nm * nm);
            int i = (tid / nm) % nm;
            int j = tid % nm;
        
            double r_tmp[nq];
        
            for (int r = 0; r < nq; ++r) {
                r_tmp[r] = s_wsp1[e * (nq * nm * nm) + r * (nm * nm) + i * nm + j];
            }

            for (int r = 0; r < nq; ++r) {
                s_wsp0[e * (nq * nm * nm) + i * (nm * nq) + j * nq + r] = r_tmp[r];
            }
        }
        __syncwarp();

        //s_wsp0(eijr) . s_basis(kr) = s_wsp1(eijk)
        f64_m8n8k4_tiled_gemm<nelmtPerBatch * nm * nm, nm, nq, Layout::RowMajor, Layout::ColMajor, Layout::RowMajor>(s_wsp0, s_basis, s_wsp1);



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