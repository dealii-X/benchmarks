#include <iostream>
#include <kernels/BK5/templated_cuda_kernels.cuh>
#include <timer.hpp>
#include <array>
#include <vector>
#include <iomanip>

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

template<typename T, unsigned int nq0, unsigned int nq1, unsigned int nq2>
void run_test(
    unsigned int nelmt, const unsigned int numThreads3D, const unsigned int ntests)
{
    //Allocation of arrays
    std::array<T, nq0 * nq0> dbasis0;
    std::array<T, nq1 * nq1> dbasis1;
    std::array<T, nq2 * nq2> dbasis2;

    std::vector<T> G  (nelmt * 6 * nq0 * nq1 * nq2, 2);
    std::vector<T> in (nelmt * nq0 * nq1 * nq2, 3);
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, 0);

    //Initialization of basis functions
    for(unsigned int i = 0u; i < nq0; i++)
    {
        for(unsigned int a = 0u; a < nq0; a++)
        {
            dbasis0[i * nq0 + a] = std::cos((T)(i * nq0 + a));
        }
    }
    for(unsigned int j = 0u; j < nq1; j++)
    {
        for(unsigned int b = 0u; b < nq1; b++)
        {
            dbasis1[j * nq1 + b] = std::cos((T)(j * nq1 + b));
        }
    }
    for(unsigned int k = 0u; k < nq2; k++)
    {
        for(unsigned int c = 0u; c < nq2; c++)
        {
            dbasis2[k * nq2 + c] = std::cos((T)(k * nq2 + c));
        }
    }

    T *d_dbasis0, *d_dbasis1, *d_dbasis2, *d_G, *d_in, *d_out;
                              
    CUDA_CHECK(cudaMalloc(&d_dbasis0, dbasis0.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis1, dbasis1.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_dbasis2, dbasis2.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_G, G.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in,  in.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, out.size() * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_dbasis0, dbasis0.data(), dbasis0.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis1, dbasis1.data(), dbasis1.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dbasis2, dbasis2.data(), dbasis2.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_G, G.data(), G.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in,  in.data(),   in.size() * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, out.data(), out.size() * sizeof(T), cudaMemcpyHostToDevice));
    


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    std::cout << std::fixed << std::setprecision(3);

    std::cout << std::left  << std::setw(15) << "Kernel"
              << std::right << std::setw(4)  << "p0"
              << std::right << std::setw(4)  << "p1"
              << std::right << std::setw(4)  << "p2"
              << std::right << std::setw(12)  << "nelmt"
              << std::right << std::setw(16) << "numThreads"
              << std::right << std::setw(16)  << "DOF"
              << std::right << std::setw(10)  << "time"
              << std::right << std::setw(8)  << "GDOF/s"
              << std::endl;

    // ------------------------- Kernel with 1D block size + Simple Map -------------------------------
    if(nq0 * nq1 * nq2 < maxThreadsPerBlock)
    {   
        const unsigned int numBlocks = numThreads3D / (nq0 * nq1 * nq2);
        unsigned int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_3D_Block_SimpleMap<T, nq0, nq1, nq2><<<numBlocks, nq0 * nq1 * nq2, ssize * sizeof(T)>>>(nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << std::left  << std::setw(15) << "1DS" 
                  << std::right << std::setw(4)  << nq0 - 1 
                  << std::right << std::setw(4)  << nq1 - 1
                  << std::right << std::setw(4)  << nq2 - 1
                  << std::right << std::setw(12) << nelmt
                  << std::right << std::setw(16) << numThreads3D
                  << std::right << std::setw(16) << nq0 * nq1 * nq2 * nelmt
                  << std::right << std::setw(10) << time
                  << std::right << std::setw(8)  << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time
                  << std::endl;
    }

    // ------------------------- Kernel with 2D block size (jk) Simple Map-------------------------------
    if(nq1 * nq2 < maxThreadsPerBlock)
    {   
        const unsigned int numBlocks = (numThreads3D / nq0) / (nq1 * nq2);
        unsigned int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {   
            Timer.start();
            BK5::Parallel::TransHexKernel_QP_2D_Block_jk_SimpleMap<T, nq0, nq1, nq2><<<numBlocks, nq1 * nq2, ssize * sizeof(T)>>>(nelmt,
                                                            d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        std::cout << std::left  << std::setw(15) << "2DS(jk)" 
                  << std::right << std::setw(4)  << nq0 - 1 
                  << std::right << std::setw(4)  << nq1 - 1
                  << std::right << std::setw(4)  << nq2 - 1
                  << std::right << std::setw(12) << nelmt
                  << std::right << std::setw(16) << numThreads3D / nq0
                  << std::right << std::setw(16) << nq0 * nq1 * nq2 * nelmt
                  << std::right << std::setw(10) << time
                  << std::right << std::setw(8)  << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time
                  << std::endl;
    }

    cudaFree(d_dbasis0); cudaFree(d_dbasis1); cudaFree(d_dbasis2); cudaFree(d_G); cudaFree(d_in); cudaFree(d_out);
}


int main(int argc, char **argv){
    const unsigned int nq0 =  4;
    const unsigned int nq1 =  4;
    const unsigned int nq2 =  4;

    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2<<18;
    unsigned int numThreads3D       = (argc > 2) ? atoi(argv[2]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int ntests             = (argc > 3) ? atoi(argv[3]) : 50u;
    
    std::cout.precision(8);
    run_test<float, nq0, nq1, nq2>(nelmt, numThreads3D, ntests);

    return 0;
}