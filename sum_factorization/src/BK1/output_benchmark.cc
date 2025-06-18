/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <fstream>
#include <kernels/BK1/cuda_kernels.cuh>
#include <kernels/BK1/kokkos_kernels.hpp>
#include <kernels/BK1/serial_kernels.hpp>
#include <Kokkos_Core.hpp>
#include <timer.hpp>
#include <math/logspace.hpp>
#include <cstdlib>
#include <string>

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

template<typename T>
std::vector<T> run_test(const unsigned int nq0, const unsigned int nq1, const unsigned int nq2, 
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2, 
    const unsigned int numThreads, const unsigned int threadsPerBlockX, const unsigned int threadsPerBlockY, 
    const unsigned int threadsPerBlockZ, const unsigned int nelmt, const unsigned int ntests)
{   
    unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
    const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

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
    std::fill(out, out + nelmt * nm0 * nm1 * nm2, (T)0.f);

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


    std::vector<T> results(5);
    // ------------------------- Kokkos Kernel ---------------------------------------------------
    {   
        std::vector<T> kokkos_results = Parallel::KokkosKernel<T>(nq0, nq1, nq2, nm0, nm1, nm2, basis0, basis1, basis2, JxW, in, out, numThreads, threadsPerBlock, nelmt, ntests);
        results[0] = kokkos_results[0];
    }


    // --------------------------Cuda Kernels ----------------------------------------------------
    T *d_basis0, *d_basis1, *d_basis2, *d_JxW, *d_in, *d_out;
    
    CUDA_CHECK(cudaMalloc(&d_basis0, nq0 * nm0 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis1, nq1 * nm1 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_basis2, nq2 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_in, nelmt * nm0 * nm1 * nm2 * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_out, nelmt * nq0 * nq1 * nq2 * sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_basis0, basis0, nm0 * nq0 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis1, basis1, nm1 * nq1 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_basis2, basis2, nm2 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_JxW, JxW, nelmt * nq0 * nq1 * nq2 * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in, in, nelmt * nm0 * nm1 * nm2 * sizeof(T), cudaMemcpyHostToDevice));

    // ------------------------- Kernel with Warp Centric Computation -------------------------------
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / 2;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        dim3 blockDim(warpSize * nelmtPerBlock);
        dim3 gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_Warp<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nm0, nm1, nm2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        results[1] = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time;

    }

    // ------------------------- Kernel with Warp Centric Computation for Q1 -------------------------------
    if(nm0 == 2 && nm1 == 2 && nm2 == 2)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        int warpSize = deviceProp.warpSize;         int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock / 2;
        int shmem = deviceProp.sharedMemPerBlock; //in byte 

        int nelmtPerBlock = (shmem / sizeof(T) - nq0 * nm0 - nq1 * nm1 - nq2 * nm2) / (2 * nq0 * nq1 * nq2);
        nelmtPerBlock = std::min(nelmtPerBlock, maxThreadsPerBlock / warpSize);

        dim3 blockDim(warpSize * nelmtPerBlock);
        dim3 gridDim(numThreads / (warpSize * nelmtPerBlock));

        const unsigned int ssize = nelmtPerBlock * 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_Warp_Q1<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nm0, nm1, nm2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        results[2] = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time;
    }


    const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2;          //shared memory dynamic size


    // ------------------------- 1D block size -------------------------------
    {   
        thrust::fill(thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2, static_cast<T>(0));
        
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D<T><<<numBlocks, std::min(nq0 * nq1 * nq2, threadsPerBlock), ssize * sizeof(T)>>>(nq0, nq1, nq2, nm0, nm1, nm2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        results[3] = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time;
    }

    // ------------------------- 3D block size -------------------------------
    {
        thrust::fill(thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2, static_cast<T>(0));

        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numBlocks);
        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nm0, nm1, nm2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();                       
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        results[4] = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time;
    }

    // ------------------------- 3D block size + SimpleMap -------------------------------
    {
        thrust::fill(thrust::device, d_out, d_out + nelmt * nm0 * nm1 * nm2, static_cast<T>(0));
        
        double time = std::numeric_limits<double>::max();
        Timer Timer;
        dim3 gridDim(numBlocks);
        dim3 blockDim(nq0, nq1, nq2);

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            Timer.start();
            Parallel::BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap<T><<<gridDim, blockDim, ssize * sizeof(T)>>>(nq0, nq1, nq2, nm0, nm1, nm2, nelmt,
                                                            d_basis0, d_basis1, d_basis2, d_JxW, d_in, d_out);
            CUDA_LAST_ERROR_CHECK();                     
            CUDA_CHECK(cudaDeviceSynchronize());
            Timer.stop();
            time = std::min(time, Timer.elapsedSeconds());
        }
        results[5] = 1.0e-9 * nelmt * nm0 * nm1 * nm2 / time;
    }
    
    cudaFree(d_basis0); cudaFree(d_basis1); cudaFree(d_basis2); cudaFree(d_JxW); cudaFree(d_in); cudaFree(d_out);
    delete[] basis0; delete[] basis1; delete[] basis2; delete[] JxW; delete[] in; delete[] out;

    return results;
}


int main(int argc, char **argv){
    unsigned int polyOrderBegin      = (argc > 1) ? atoi(argv[1]) : 1;
    unsigned int polyOrderEnd        = (argc > 2) ? atoi(argv[2]) : 8;
    unsigned int dof_ExpBegin        = (argc > 3) ? atoi(argv[3]) : 4;
    unsigned int dof_ExpEnd          = (argc > 4) ? atoi(argv[4]) : 7;
    unsigned int dof_NumSample       = (argc > 5) ? atoi(argv[5]) : 45;
    unsigned int nElmtPerBlock       = (argc > 6) ? atoi(argv[6]) : 2;     // Number of Elements Per Thread Block (block-stride mechanism)
    unsigned int ntests              = (argc > 7) ? atoi(argv[7]) : 30u;
    std::string  outPath             = (argc > 8) ? std::string{argv[8]} : std::string{"./outputs"};

    const unsigned int numKernel = 6; //kokkos, cudaWarp, cudaWarpQ1, cuda1D, cuda3D, cuda3DS
    const unsigned int nmBegin = polyOrderBegin + 1;
    const unsigned int nmEnd = polyOrderEnd + 1;


    Kokkos::initialize(argc, argv);

    auto dofs = logspace<float>(dof_ExpBegin, dof_ExpEnd, dof_NumSample);

    //-------------------- allocate 3D vector for outputs-----------------
    std::vector<std::vector<std::vector<float>>> results(dofs.size(), std::vector<std::vector<float>>(nmEnd - nmBegin + 1, std::vector<float>(numKernel, 0.0f)));

    unsigned int nelmt;          
    for(unsigned int d = 0; d < dofs.size(); ++d)
    {
        for(unsigned int nm = nmBegin; nm <= nmEnd; ++nm)
        {
            nelmt = dofs[d] / (float)(nm * nm * nm);
            results[d][nm-nmBegin] = run_test<float>(nm + 1, nm + 1, nm + 1, nm, nm, nm, nelmt * (nm + 1) * (nm + 1) * (nm + 1) / nElmtPerBlock, 
                                      nm + 1, nm + 1, nm + 1, nelmt, ntests);
        }
    }

    //--------------------write 3D outputs to the file-----------------------
    std::ofstream kokkosFile; kokkosFile.open(outPath + std::string{"/kokkos.txt"}, std::ios::trunc);  if(!kokkosFile){std::cerr << "Failed to open kokkos.txt.\n"; return 1;}
    std::ofstream cudaWarpFile; cudaWarpFile.open(outPath + std::string{"/cudaWarp.txt"}, std::ios::trunc);  if(!cudaWarpFile){std::cerr << "Failed to open cudaWarp.txt.\n"; return 1;}
    std::ofstream cudaWarpQ1File; cudaWarpQ1File.open(outPath + std::string{"/cudaWarpQ1.txt"}, std::ios::trunc);  if(!cudaWarpQ1File){std::cerr << "Failed to open cudaWarpQ1.txt.\n"; return 1;}
    std::ofstream cudaOneDFile; cudaOneDFile.open(outPath + std::string{"/cuda1D.txt"}, std::ios::trunc);  if(!cudaOneDFile){std::cerr << "Failed to open cuda1D.txt.\n"; return 1;}
    std::ofstream cudaThreeDFile; cudaThreeDFile.open(outPath + std::string{"/cuda3D.txt"}, std::ios::trunc);  if(!cudaThreeDFile){std::cerr << "Failed to open cuda3D.txt.\n"; return 1;}
    std::ofstream cudaThreeDSFile; cudaThreeDSFile.open(outPath + std::string{"/cuda3DS.txt"}, std::ios::trunc);  if(!cudaThreeDSFile){std::cerr << "Failed to open cuda3DS.txt.\n"; return 1;}

    for(unsigned int d = 0; d < dofs.size(); ++d){
        kokkosFile      << dofs[d] << " ";
        cudaWarpFile    << dofs[d] << " ";
        cudaWarpQ1File  << dofs[d] << " ";
        cudaOneDFile    << dofs[d] << " ";
        cudaThreeDFile  << dofs[d] << " ";
        cudaThreeDSFile << dofs[d] << " ";

        for(unsigned int nm = nmBegin; nm <= nmEnd; ++nm){
            kokkosFile << results[d][nm-nmBegin][0] <<" ";
            cudaWarpFile << results[d][nm-nmBegin][1] <<" ";
            cudaWarpQ1File << results[d][nm-nmBegin][2] <<" ";
            cudaOneDFile << results[d][nm-nmBegin][3] <<" ";
            cudaThreeDFile << results[d][nm-nmBegin][4] <<" ";
            cudaThreeDSFile << results[d][nm-nmBegin][5] <<" ";
        }
        kokkosFile << "\n"; cudaWarpFile << "\n"; cudaWarpQ1File << "\n"; cudaOneDFile << "\n"; cudaThreeDFile << "\n"; cudaThreeDSFile << "\n";
    }
    kokkosFile.close(); cudaWarpFile.close(); cudaWarpQ1File.close(); cudaOneDFile.close(); cudaThreeDFile.close(); cudaThreeDSFile.close();

    Kokkos::finalize();
    return 0;
}