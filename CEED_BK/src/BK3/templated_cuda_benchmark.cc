#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <kernels/BK3/templated_cuda_kernels.cuh>
#include <timer.hpp>
#include <benchmark_printer.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUDA_LAST_ERROR_CHECK()                                               \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Last Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
              const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;

    //Allocation of arrays
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];
    T* weights = new T[nq];

    T* coord_q = new T[nelmt * 3 * nquad];
    T* G = new T[nelmt * 6 * nquad];
    T* in = new T[nelmt * nm*nm*nm];
    T* out = new T[nelmt * nm*nm*nm];

    //Initialize the input and output arrays
    for(unsigned int i = 0; i < nelmt * nm * nm * nm; ++i)
        in[i] = std::sin((T)i); 

    std::fill(out, out + nelmt * nm*nm*nm, (T)0.0);
    std::fill(weights, weights + nq, (T)1.0);

    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[p * nm + i] = std::cos((T)(p * nm + i));
        }
    }

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    for(unsigned int e = 0; e < nelmt; ++e){
        unsigned int coord_base = e * 3 * nquad;
        unsigned int xbase = coord_base;
        unsigned int ybase = coord_base + nquad;
        unsigned int zbase = coord_base + 2 * nquad;
        
        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){
                    unsigned int idx = p * nq * nq + q * nq + r;
                    coord_q[xbase + idx] = (T)p + 0.1 * (T)q + 0.1 * (T)r;
                    coord_q[ybase + idx] = 0.1 * (T)p + (T)q + 0.1 * (T)r;
                    coord_q[zbase + idx] = 0.1 * (T)p + 0.1 * (T)q + (T)r;
                }
            }
        }
    }

    //Compute G from coord_q using the same J -> C -> detJ formulation as OTF
    for(unsigned int e = 0; e < nelmt; ++e){
        unsigned int coord_base = e * 3 * nquad;
        unsigned int xbase = coord_base;
        unsigned int ybase = coord_base + nquad;
        unsigned int zbase = coord_base + 2 * nquad;
        unsigned int Gbase = e * 6 * nquad;

        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){

                    T J00 = 0.0, J01 = 0.0, J02 = 0.0;
                    T J10 = 0.0, J11 = 0.0, J12 = 0.0;
                    T J20 = 0.0, J21 = 0.0, J22 = 0.0;

                    for(unsigned int n = 0; n < nq; ++n){
                        const T x_r = coord_q[xbase + n * nq * nq + q * nq + r];
                        const T x_s = coord_q[xbase + p * nq * nq + n * nq + r];
                        const T x_t = coord_q[xbase + p * nq * nq + q * nq + n];

                        J00 += dbasis[n * nq + p] * x_r;
                        J01 += dbasis[n * nq + q] * x_s;
                        J02 += dbasis[n * nq + r] * x_t;

                        const T y_r = coord_q[ybase + n * nq * nq + q * nq + r];
                        const T y_s = coord_q[ybase + p * nq * nq + n * nq + r];
                        const T y_t = coord_q[ybase + p * nq * nq + q * nq + n];

                        J10 += dbasis[n * nq + p] * y_r;
                        J11 += dbasis[n * nq + q] * y_s;
                        J12 += dbasis[n * nq + r] * y_t;

                        const T z_r = coord_q[zbase + n * nq * nq + q * nq + r];
                        const T z_s = coord_q[zbase + p * nq * nq + n * nq + r];
                        const T z_t = coord_q[zbase + p * nq * nq + q * nq + n];

                        J20 += dbasis[n * nq + p] * z_r;
                        J21 += dbasis[n * nq + q] * z_s;
                        J22 += dbasis[n * nq + r] * z_t;
                    }

                    const T C00 = J11 * J22 - J12 * J21;
                    const T C01 = J02 * J21 - J01 * J22;
                    const T C02 = J01 * J12 - J02 * J11;

                    const T C10 = J12 * J20 - J10 * J22;
                    const T C11 = J00 * J22 - J02 * J20;
                    const T C12 = J02 * J10 - J00 * J12;

                    const T C20 = J10 * J21 - J11 * J20;
                    const T C21 = J01 * J20 - J00 * J21;
                    const T C22 = J00 * J11 - J01 * J10;

                    const T detJ = J00 * C00 + J01 * C10 + J02 * C20;
                    const T scale = (weights[p] * weights[q] * weights[r]) / detJ;

                    const T G00 = scale * (C00 * C00 + C01 * C01 + C02 * C02);
                    const T G01 = scale * (C00 * C10 + C01 * C11 + C02 * C12);
                    const T G02 = scale * (C00 * C20 + C01 * C21 + C02 * C22);
                    const T G11 = scale * (C10 * C10 + C11 * C11 + C12 * C12);
                    const T G12 = scale * (C10 * C20 + C11 * C21 + C12 * C22);
                    const T G22 = scale * (C20 * C20 + C21 * C21 + C22 * C22);

                    unsigned int idx = p * nq * nq + q * nq + r;

                    G[Gbase + 0 * nquad + idx] = G00;
                    G[Gbase + 1 * nquad + idx] = G01;
                    G[Gbase + 2 * nquad + idx] = G02;
                    G[Gbase + 3 * nquad + idx] = G11;
                    G[Gbase + 4 * nquad + idx] = G12;
                    G[Gbase + 5 * nquad + idx] = G22;
                }
            }
        }
    }

    // Allocate and transfer device memory
    T *d_basis, *d_dbasis, *d_G, *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_basis, nm * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis, nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, nelmt * 6 * nquad * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nm * nm * nm * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nm * nm * nm * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_basis, basis, nm * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis, dbasis, nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, nelmt * 6 * nquad * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nm * nm * nm * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nm * nm * nm * sizeof(T), cudaMemcpyHostToDevice));

    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- 2D Block(pq) Kernel ---------------------------------------------------
    {   
        unsigned int ssize = nm*nq + nq*nq + 4 * nelmtPerBatch * nq * nq * nq; 

        double time = std::numeric_limits<double>::max();
        Timer timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            timer.start();
            BK3::Parallel::LaplaceOperator<T, nq><<<numBlocks, threadsPerBlock, ssize * sizeof(T)>>>(
                nelmt, nelmtPerBatch, d_basis, d_dbasis, d_G, d_in, d_out
            );
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            time = std::min(time, timer.elapsedSeconds());
        }
        
        T sum = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nm * nm * nm,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );

        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt; uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        
        T DOFs = 1.0e-9 * nDOF / time;
        T bw = 1.0e-9 *(2 * nDOF + 6*nQuad) * sizeof(T) / time;
        printer("BK3", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    CUDA_CHECK(cudaFree(d_basis)); 
    CUDA_CHECK(cudaFree(d_dbasis)); 
    CUDA_CHECK(cudaFree(d_G)); 
    CUDA_CHECK(cudaFree(d_in)); 
    CUDA_CHECK(cudaFree(d_out));
    
    delete[] basis; delete[] dbasis; delete[] weights; delete[] coord_q; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u; unsigned int nq = p + 2;
    size_t nelmt                   = (argc > 2) ? std::stoull(argv[2]) : (1ULL << 16);
    
    unsigned int nelmtPerBatch     = (argc > 3) ? atoi(argv[3]) : std::max(1UL, shmemPerBlock / (4 * nq * nq * nq) / sizeof(T));
    unsigned int numBlocks         = (argc > 4) ? atoi(argv[4]) : std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

    threadsPerBlock                = (argc > 5) ? atoi(argv[5]) : threadsPerBlock;
    unsigned int ntests            = (argc > 6) ? atoi(argv[6]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 10: run_test<T, 10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    return 0;
}