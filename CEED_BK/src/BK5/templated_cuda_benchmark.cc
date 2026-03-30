#include <iostream>
#include <kernels/BK5/templated_cuda_kernels.cuh>
#include <timer.hpp>
#include <benchmark_printer.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUDA_LAST_ERROR_CHECK()                                                   \
    do {                                                                          \
        cudaError_t err = cudaGetLastError();                                     \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA Last Error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{
    //Allocation of arrays
    T* dbasis = new T[nq * nq];
    T* G = new T[nelmt * 6 * nq * nq * nq];
    T* in = new T[nelmt * nq * nq * nq];
    T* out = new T[nelmt * nq * nq * nq];


    //Initialize the input and output arrays
    std::fill(G,   G + nelmt * nq * nq * nq * 6, (T)2.0f);
    std::fill(in,  in + nelmt * nq * nq * nq,   (T)3.0f);
    std::fill(out, out + nelmt * nq * nq * nq, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int a = 0u; a < nq; a++)
        {
            dbasis[i * nq + a] = std::cos((T)(i * nq + a));
        }
    }

    T *d_dbasis, *d_G, *d_in, *d_out;
                              
    CUDA_CHECK(cudaMalloc(&d_dbasis, nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, nelmt * nq * nq * nq * 6 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nq * nq * nq * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nq * nq * nq * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_dbasis, dbasis, nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G, nelmt * nq * nq * nq * 6 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nq * nq * nq * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nq * nq * nq * sizeof(T), cudaMemcpyHostToDevice));

    int device;   cudaGetDevice(&device);   cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    

    BenchmarkPrinter<T> printer;
    printer.print_header();


    // ------------------------- Kernel with 2D block size (jk)-------------------------------
    {   
        unsigned int ssize = nq * nq + 4 * nelmtPerBatch * nq * nq * nq;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::LaplaceOperator<T, nq><<<numBlocks, threadsPerBlock, ssize * sizeof(T)>>>(nelmt, nelmtPerBatch,
                                                        d_dbasis, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        
        
        T sum = thrust::transform_reduce(
            thrust::device, d_out, d_out + nelmt * nq * nq * nq,
            thrust::square<T>(), (T)0.0,
            thrust::plus<T>()
        );


        uint64_t nDOF = (uint64_t)nq * nq * nq * nelmt; uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T DOFs = 1.0e-9 * nDOF / time;
        T bw = 1.0e-9 *(2 * nDOF + 6*nQuad) * sizeof(T) / time;
        printer("BK5", nq - 1, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }


    cudaFree(d_dbasis); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
    delete[] dbasis; delete[] G; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = float;
    int shmemPerBlock = 10800;

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 3u; unsigned int nq = p + 1;
    unsigned int nelmt             = (argc > 2) ? atoi(argv[2]) : 2 << 18;
    
    unsigned int nelmtPerBatch     = shmemPerBlock / (4 * nq * nq * nq) / sizeof(T);    if(nelmtPerBatch == 0) nelmtPerBatch = 1;
    unsigned int numBlocks         = (argc > 3) ? atoi(argv[3]) : (nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2;

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);
                 threadsPerBlock   = ((threadsPerBlock + 31) / 32) * 32;

    threadsPerBlock                = (argc > 4) ? atoi(argv[4]) : threadsPerBlock;
    unsigned int ntests            = (argc > 5) ? atoi(argv[5]) : 10u;

    std::cout.precision(8);
        switch(nq) {
        case 2: run_test<T, 2>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    return 0;
}