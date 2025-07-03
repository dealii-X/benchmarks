/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120  // or 120, 100, depending on what you want
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>  // or cl2.hpp if you prefer

#include <iostream>
#include <fstream>
#include <execution>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>

#include "timer.hpp"


// Take a list of key value pairs to be used as compile-time options
template <typename T>
std::string buildOpenCLDefines(const std::unordered_map<std::string, T>& defines) {
    std::string options;
    for (const auto& [key, value] : defines) {
        options += "-D" + key + "=" + std::to_string(value) + " ";
    }
    return options;
}


template <typename Container>
auto norm2(const Container& data) -> typename Container::value_type {
    using T = typename Container::value_type;
    return std::transform_reduce(
        std::execution::par_unseq,
        data.begin(), data.end(),
        T{0},
        std::plus<>(),
        [](T val) { return val * val; }
    );
}

template<typename T = float, bool static_size = false>
void run_test(
    cl::Context &context, cl::CommandQueue &queue, const std::string &kernelName,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2, 
    const unsigned int numThreads, 
    const unsigned int threadsPerBlockX,
    const unsigned int threadsPerBlockY,
    const unsigned int threadsPerBlockZ, 
    const unsigned int nelmt, const unsigned int ntests)
{
    unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
    const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

    //Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, (T)0.0);

    //Initialization of basis functions
    for(int p = 0; p < nq0; p++) {
        for(int i = 0; i < nm0; i++) {
            basis0[p * nm0 + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for(int q = 0; q < nq1; q++) {
        for(int j = 0; j < nm1; j++) {
            basis1[q * nm1 + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for(int r = 0; r < nq2; r++) {
        for(int k = 0; k < nm2; k++) {
            basis2[r * nm2 + k] = std::cos((T)(r * nm2 + k));
        }
    }

    const size_t size_JxW = nelmt * nq0 * nq1 * nq2;
    const size_t size_in = nelmt * nm0 * nm1 * nm2;
    const size_t size_out = nelmt * nq0 * nq1 * nq2;
    const size_t size_basis0 = nq0*nm0;
    const size_t size_basis1 = nq1*nm1;
    const size_t size_basis2 = nq2*nm2;

    const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2; //shared memory dynamic size

    const size_t sharedMemSize = ssize * sizeof(T);

    cl::Buffer d_basis0(context,CL_MEM_READ_ONLY, sizeof(T) * size_basis0);
    cl::Buffer d_basis1(context,CL_MEM_READ_ONLY, sizeof(T) * size_basis1);
    cl::Buffer d_basis2(context,CL_MEM_READ_ONLY, sizeof(T) * size_basis2);
    cl::Buffer d_JxW(context,CL_MEM_READ_ONLY, sizeof(T) * size_JxW);
    cl::Buffer d_in(context,CL_MEM_READ_ONLY, sizeof(T) * size_in);
    cl::Buffer d_out(context,CL_MEM_READ_WRITE, sizeof(T) * size_out);

    queue.enqueueWriteBuffer(d_basis0, CL_TRUE, 0, sizeof(T) * size_basis0, basis0.data());
    queue.enqueueWriteBuffer(d_basis1, CL_TRUE, 0, sizeof(T) * size_basis1, basis1.data());
    queue.enqueueWriteBuffer(d_basis2, CL_TRUE, 0, sizeof(T) * size_basis2, basis2.data());
    queue.enqueueWriteBuffer(d_JxW, CL_TRUE, 0, sizeof(T) * size_JxW, JxW.data());
    queue.enqueueWriteBuffer(d_in, CL_TRUE, 0, sizeof(T) * size_in, in.data());

    // Kernel compilation

    // 7. Read kernel source code from file
    std::ifstream kernelFile(
        static_size ? 
            "./include/kernels/BK1/opencl_kernels_static.cl" : 
            "./include/kernels/BK1/opencl_kernels.cl");

    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file.\n";
        return;
    }

    std::string source(std::istreambuf_iterator<char>(kernelFile),
                       (std::istreambuf_iterator<char>()));

    std::unordered_map<std::string, size_t> defines = {
       {"NM0", (size_t) nm0},
       {"NM1", (size_t) nm1},
       {"NM2", (size_t) nm2},
       {"NQ0", (size_t) nq0},
       {"NQ1", (size_t) nq1},
       {"NQ2", (size_t) nq2}
    };

    std::string buildOptions = static_size ? buildOpenCLDefines(defines) : "";

    // By default, the kernel assume float is used
    if constexpr (std::is_same_v<T, double>) {
        buildOptions += " -DT=double";
    }

    // 8. Build program from source
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    cl::Program program(context, source);
    try {
        program.build({device},buildOptions);
    } catch (const cl::Error&) {
        // If build fails, print build log
        std::cerr << "Build failed:\n" 
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        return;
    }

    cl::Kernel kernel(program, kernelName);

    // In the static kernel, the small integer sizes have
    // been passed directly to the OpenCL JIT compiler

    if constexpr (static_size) {
        kernel.setArg(0, nelmt);
        kernel.setArg(1, d_basis0);
        kernel.setArg(2, d_basis1);
        kernel.setArg(3, d_basis2);
        kernel.setArg(4, d_JxW);
        kernel.setArg(5, d_in);
        kernel.setArg(6, d_out);
        kernel.setArg(7, cl::Local(sharedMemSize));        
    } else {
        kernel.setArg(0, nq0);
        kernel.setArg(1, nq1);
        kernel.setArg(2, nq2);
        kernel.setArg(3, nm0);
        kernel.setArg(4, nm1);
        kernel.setArg(5, nm2);
        kernel.setArg(6, nelmt);
        kernel.setArg(7, d_basis0);
        kernel.setArg(8, d_basis1);
        kernel.setArg(9, d_basis2);
        kernel.setArg(10, d_JxW);
        kernel.setArg(11, d_in);
        kernel.setArg(12, d_out);
        kernel.setArg(13, cl::Local(sharedMemSize));
    }

    const size_t globalSize = numThreads;
    const size_t localSize = std::min(nq0 * nq1 * nq2, threadsPerBlock);

    double time_cl = std::numeric_limits<double>::max();

    cl::NDRange global, local;
    if (kernelName == "BwdTransHexKernel_QP_1D") {
//    dim3 gridDim(numBlocks)
//    dim3 blockDim(std::min(nq0 * nq1 * nq2, threadsPerBlock))
        global = cl::NDRange(numThreads);
        local = cl::NDRange(std::min(nq0 * nq1 * nq2, threadsPerBlock));
    } else if (kernelName == "BwdTransHexKernel_QP_1D_3D_BLOCKS") {
//        dim3 gridDim(numBlocks);
//        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));
        global = cl::NDRange(numThreads);
        local = cl::NDRange(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));
    } else if (kernelName == "BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap") {
//        dim3 gridDim(numBlocks);
//        dim3 blockDim(nq0, nq1, nq2);
        global = cl::NDRange(numThreads);
        local = cl::NDRange(nq0, nq1, nq2);      
    }

    Timer clTimer;
    for (unsigned int t = 0u; t < ntests; ++t)
    {
        clTimer.start();
        
        queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,local);
        queue.finish();

        clTimer.stop();
        time_cl = std::min(time_cl, clTimer.elapsedSeconds());
    }

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) { return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed; };


    // C++23
    // std::println("OpenCL -> nelmt = {} GDoF/s = {}", nelmt, dof_rate(time_cl));

    // kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()

    std::cout << kernelName << '\n';
    std::cout << "OpenCL "
              << "nelmt = " << nelmt 
              << " GDoF/s = " << dof_rate(time_cl) 
              << '\n';

    // 1. Allocate host buffer to receive data
    std::vector<T> host_out(size_out);

    // 2. Copy data from device buffer d_out to host
    queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(T) * host_out.size(), host_out.data());

    // Evaluate the norm in double precision
    double normSqr = 0;
    for (auto h : host_out) {
        normSqr += ((double) h) * ((double) h);
    }

    //const T normSqr = norm2(host_out);
    //std::println("OpenCL kernel norm = {}", std::sqrt(normSqr));

    std::cout << "OpenCL kernel norm = " << std::sqrt(normSqr) << '\n';

}


int main(int argc, char **argv){

    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    unsigned int numThreads         = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    unsigned int ntests             = (argc > 9) ? atoi(argv[9]) : 50u;


    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    // FIXME: initialize OpenCL context

    try {
        // 1. Get all platforms (e.g., NVIDIA, Intel, AMD)
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "No OpenCL platforms found.\n";
            return 1;
        }

        // 2. Select the first platform
        cl::Platform platform = platforms[0];
        std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        // 3. Get GPU devices from platform (fallback to CPU if needed)
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            std::cout << "No GPU devices found; trying CPU devices...\n";
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                std::cerr << "No devices found.\n";
                return 1;
            }
        }

        // 4. Select the first device
        cl::Device device = devices[0];
        std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        // 5. Create context with the selected device
        cl::Context context(device);

        // 6. Create command queue (enable profiling if needed)
        cl::CommandQueue queue(context, device);

        std::cout << "--- Dynamic size ---\n";
        run_test<float,false>(context, queue, "BwdTransHexKernel_QP_1D",           
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
        run_test<float,false>(context, queue, "BwdTransHexKernel_QP_1D_3D_BLOCKS",
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
        run_test<float,false>(context, queue, "BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap",
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);


        std::cout << "--- Static size ---\n";
        run_test<float,true>(context, queue, "BwdTransHexKernel_QP_1D",           
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
        run_test<float,true>(context, queue, "BwdTransHexKernel_QP_1D_3D_BLOCKS",
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);
        run_test<float,true>(context, queue, "BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap",
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);


    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }

    return 0;
}