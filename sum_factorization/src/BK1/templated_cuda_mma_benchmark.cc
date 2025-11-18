#include <iostream>
#include <kernels/BK1/templated_cuda_mma_kernels.cuh>
#include <timer.hpp>
#include <benchmark_printer.hpp>


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


template<typename T, const unsigned int nq0, const unsigned int nq1, const unsigned int nq2>
void run_test(
    const unsigned int numThreads, const unsigned int nelmt, const unsigned int ntests)
{
    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    //Allocation of arrays
    T* basis0 = new T[nm0 * nq0];
    T* basis1 = new T[nm1 * nq1];
    T* basis2 = new T[nm2 * nq2];
    T* JxW = new T[nelmt * nq0 * nq1 * nq2];
    T* in = new T[nelmt * nm0 * nm1 * nm2];
    T* out = new T[nelmt * nm0 * nm1 * nm2];


    //Initialize the input and output arrays
    std::fill(JxW, JxW + nelmt * nq0 * nq1 * nq2, (T)1.0f);
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (T)3.0f);
    std::fill(out, out + nelmt * nm0 * nm1 * nm2, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq0; p++)
    {
        for(unsigned int i = 0u; i < nm0; i++)
        {
            basis0[p * nm0 + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for(unsigned int q = 0u; q < nq1; q++)
    {
        for(unsigned int j = 0u; j < nm1; j++)
        {
            basis1[q * nm1 + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for(unsigned int r = 0u; r < nq2; r++)
    {
        for(unsigned int k = 0u; k < nm2; k++)
        {
            basis2[r * nm2 + k] = std::cos((T)(r * nm2 + k));
        }
    }

    T *d_basis0, *d_basis1, *d_basis2, *d_JxW, *d_in, *d_out;
                              
    CUDA_CHECK(cudaMalloc(&d_basis0, nq0 * nm0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis1, nq1 * nm1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis2, nq2 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nm0 * nm1 * nm2 * sizeof(T)));


    CUDA_CHECK(cudaMemcpy(d_basis0, basis0, nm0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis1, basis1, nm1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis2, basis2, nm2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_JxW, JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));


    BenchmarkPrinter printer;
    printer.print_header();

    // ------------------------- Kernel with Single Warp per Element -------------------------------
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int warpSize = deviceProp.warpSize;

    const unsigned int numBlocks = numThreads / warpSize;

    const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

    double time = std::numeric_limits<double>::max();
    Timer Timer;

    for (unsigned int t = 0u; t < ntests; ++t)
    {   
        Timer.start();
        BK1::Parallel::BwdTransHexKernel_mma<T, 8, 8, 4, nq0, nq1, nq2><<<numBlocks, warpSize, ssize * sizeof(T)>>>(nelmt,
                                                        d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
        CUDA_LAST_ERROR_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
        Timer.stop();
        time = std::min(time, Timer.elapsedSeconds());
    }
    
    printer("mma", nq0 - 2, nq1 - 2, nq2 - 2, nelmt, numBlocks * warpSize, nm0 * nm1 * nm2 * nelmt, time, 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time);
}



int main(int argc, char **argv){

    constexpr unsigned int nq0      =  4u;
    constexpr unsigned int nq1      =  4u;
    constexpr unsigned int nq2      =  4u;
    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    unsigned int numThreads         = (argc > 2) ? atoi(argv[2]) : nelmt * 32 / 4;
    unsigned int ntests             = (argc > 3) ? atoi(argv[3]) : 50u;

    std::cout.precision(8);
    run_test<double, nq0, nq1, nq2>(numThreads, nelmt, ntests);

    return 0;
}