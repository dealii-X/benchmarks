#ifndef BK1_CUDA_MMA_KERNELS_CUH
#define BK1_CUDA_MMA_KERNELS_CUH


namespace BK1{
namespace Parallel{

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


template<int m, int n, int k, Layout Layout_A, Layout Layout_B, Layout Layout_C, int num_batch, int M, int N, int K>
__device__ void batched_tiled_gemm(double *s_batched_A, double *s_B, double *s_batched_C)
{
    const int tid = threadIdx.x;
    const int laneid = tid % warpSize;

    constexpr int num_tiles_m = (M + m - 1) / m;
    constexpr int num_tiles_n = (N + n - 1) / n;
    constexpr int num_tiles_k = (K + k - 1) / k;

    double r_b[K][N] = {0};

    auto s_B_view = matrixView<double, Layout_B, K, N>(s_B);

    //copy s_B from shared memory to register
    {
    int base_row = laneid % 4;
    int base_col = laneid >> 2;
    
    int row = base_row;
    int col = base_col;
    
    for(int i = 0; i < num_tiles_k; i++){
        row = base_row + i * k;     if(row >= K) break;
        for(int j = 0; j < num_tiles_n; j++){
            col = base_col + j * n;     if(col >= N) break;
            r_b[i][j] = s_B_view(row, col);
        }
    }
    __syncwarp();
    }

    //batch iteration
    for(int batch_id = 0; batch_id < num_batch; batch_id++)
    {

        auto current_s_A_view = matrixView<double, Layout_A, M, K>(s_batched_A + batch_id * (M * K));

        double r_a[M][K] = {0};
        double r_c[M][N][2] = {0};
        
        //copy s_A from shared memory to register
        {
        int base_row = laneid >> 2;
        int base_col = laneid % 4;

        int row = base_row;
        int col = base_col;

        for(int i = 0; i < num_tiles_m; i++){
            row = base_row + i * m;      if(row >= M) break;
            for(int j = 0; j < num_tiles_k; j++){
                col = base_col + j * k;      if(col >= K) break;
                r_a[i][j] = current_s_A_view(row, col);
            }
        }
        __syncwarp();
        }

        //tiled GEMM
        for(int i = 0; i < num_tiles_m; i++){
            for(int j = 0; j < num_tiles_n; j++){
                for(int t = 0; t < num_tiles_k; t++)
                {
                    //Perform mma
                    asm (
                        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                        "{%0, %1}, {%2}, {%3}, {%0, %1}; \n"
                        :"+d"(r_c[i][j][0]), "+d"(r_c[i][j][1])
                        :"d"(r_a[i][t]),
                         "d"(r_b[t][j])
                    );
                }
            }
        }

        auto current_s_C_view = matrixView<double, Layout_C, M, N>(s_batched_C + batch_id * (M * N));

        {
        //copy from register to shared memory s_C
        int base_row = laneid >> 2;
        int base_col = (laneid % 4) * 2;
        
        int row = base_row;
        int col = base_col;

        for(int i = 0; i < num_tiles_m; ++i){
            row = base_row + i * m;     if(row >= M) break;
            for(int j = 0; j < num_tiles_n; ++j){
                col = base_col + j * n;     if(col >= N) break;
                current_s_C_view(row, col) = r_c[i][j][0];

                col += 1;   if(col >= N) break;
                current_s_C_view(row, col) = r_c[i][j][1];
            }
        }
        __syncwarp();
        }
    }
}

template<typename T, int m, int n, int t, unsigned int nq0, unsigned int nq1, unsigned int nq2>
void __global__ BwdTransHexKernel_mma(
    const unsigned int nelmt, const T *__restrict__ d_basis0, const T *__restrict__ d_basis1,
    const T *__restrict__ d_basis2, const T *__restrict__ d_JxW, const T *__restrict__ d_in, T *__restrict__ d_out) 
{
    constexpr unsigned int nm0 = nq0 - 1;
    constexpr unsigned int nm1 = nq1 - 1;
    constexpr unsigned int nm2 = nq2 - 1;

    int laneId = threadIdx.x % warpSize;      // position of thread in warp

    extern __shared__ T shared[];
    T *s_basis0 = shared;
    T *s_basis1 = s_basis0 + nm0 * nq0;
    T *s_basis2 = s_basis1 + nm1 * nq1;
    T *s_wsp0 = s_basis2 + nm2 * nq2;
    T *s_wsp1 = s_wsp0 + nq0 * nq1 * nq2;

    //copy to shared memory
    for(unsigned int tid = threadIdx.x; tid < nq0 * nm0; tid += blockDim.x)
    {
        s_basis0[tid] = d_basis0[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq1 * nm1; tid += blockDim.x)
    {
        s_basis1[tid] = d_basis1[tid];
    }

    for(unsigned int tid = threadIdx.x; tid < nq2 * nm2; tid += blockDim.x)
    {
        s_basis2[tid] = d_basis2[tid];
    }

    unsigned int e = blockIdx.x;

    while(e < nelmt)
    {
        //step-1 : Copy from in to the wsp0
        for(unsigned int tid = threadIdx.x; tid < nm0 * nm1 * nm2; tid += blockDim.x)
        {
            s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
        }
        __syncwarp();

        //step-2 : direction 0      s_wsp0(ijk) s_basis0(pi) -> s_wsp1(pjk)
        for(unsigned int tid = laneId; tid < nq0 * nm1 * nm2; tid += warpSize)
        {
            int p = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int i = 0; i < nm0; ++i)
            {
                tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
            }
            s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-3 : direction 1      s_wsp1(pjk) s_basis1(qj) -> s_wsp0(pkq)
        batched_tiled_gemm<m, n, t, Layout::ColMajor, Layout::ColMajor, Layout::RowMajor, nq0, nm2, nq1, nm1>(s_wsp1, s_basis1, s_wsp0);

        //step-4 : direction 2      s_wsp0(pkq) s_basis2(rk) -> s_wsp1(pqr)
        batched_tiled_gemm<m, n, t, Layout::ColMajor, Layout::ColMajor, Layout::RowMajor, nq0, nq1, nq2, nm2>(s_wsp0, s_basis2, s_wsp1);

        //step-5 : Multiply with weights and determinant of Jacobi
        for(unsigned int tid = laneId; tid < nq0 * nq1 * nq2; tid += warpSize)
        {
            int p = tid / (nq1 * nq2);
            int q = (tid % (nq1 * nq2)) / nq2;
            int r = tid % nq2;

            s_wsp1[p * nq1 * nq2 + q * nq2 + r] *= d_JxW[e * nq0 * nq1 * nq2 + r * nq0 * nq1 + q * nq0 + p];
        }
        __syncwarp();

        //step-6 : direction 2      s_wsp1(pqr) s_basis2(rk) -> s_wsp0(pqk)
        batched_tiled_gemm<m, n, t, Layout::RowMajor, Layout::RowMajor, Layout::RowMajor, nq0, nq1, nm2, nq2>(s_wsp1, s_basis2, s_wsp0);

        //step-7 : direction 1      s_wsp0(pqk) s_basis1(qj) -> s_wsp1(pkj)
        batched_tiled_gemm<m, n, t, Layout::ColMajor, Layout::RowMajor, Layout::RowMajor, nq0, nm2, nm1, nq1>(s_wsp0, s_basis1, s_wsp1);

        //step-8 : direction 0      s_wsp1(pkj) s_basis2(pi) -> s_wsp0(ijk)
        for(unsigned int tid = laneId; tid < nm0 * nm1 * nm2; tid += warpSize)
        {
            int i = tid / (nm1 * nm2);
            int j = (tid % (nm1 * nm2)) / nm2;
            int k = tid % nm2;

            T tmp = 0.0;
            for(unsigned int p = 0; p < nq0; ++p)
            {
                tmp += s_wsp1[p * nm1 * nm2 + k * nm1 + j] * s_basis0[p * nm0 + i];
            }
            s_wsp0[i * nm1 * nm2 + j * nm2 + k] = tmp;
        }
        __syncwarp();

        //step-9 : Copy wsp0 to out
        for(unsigned int tid = laneId; tid < nm0 * nm1 * nm2; tid += warpSize)
        {
            d_out[e * nm0 * nm1 * nm2 + tid] = s_wsp0[tid];
        }
        __syncwarp();

        e += gridDim.x;
    }   
}

} //namespace Parallel
} //namespace BK1

#endif   //BK1_CUDA_MMA_KERNELS_CUH






    