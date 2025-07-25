// opencl_benchmarks.cc


#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120  // or 120, 100, depending on what you want
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>  // or cl2.hpp if you prefer

#include <iostream>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <numeric>

#include "timer.hpp"


// Take a list of key value pairs to be used as compile-time options
template <typename T>
std::string buildOpenCLDefines(const std::unordered_map<std::string, T>& defines) {
    std::string options;
    for (const auto& [key, value] : defines) {
        options += "-D " + key + "=" + std::to_string(value) + " ";
    }
    return options;
}


std::string loadKernelSource(bool static_size, int& status, bool verbose = false) {

    std::string filename = static_size ? "opencl_kernels_static.cl" : "opencl_kernels.cl";
    std::ifstream kernelFile;
    std::string baseDir;

    // 1. Try CL_KERNEL_DIR
    if (const char* envDir = std::getenv("CL_KERNEL_DIR")) {
        baseDir = envDir;
        std::string fullPath = baseDir + "/" + filename;
        kernelFile.open(fullPath);
        if (kernelFile.is_open() && verbose) {
            std::cout << "Loaded kernel from CL_KERNEL_DIR: " << fullPath << "\n";
        }
    }

    // 2. Fallback path
    if (!kernelFile.is_open()) {
        baseDir = "./src";
        std::string fullPath = baseDir + "/" + filename;
        kernelFile.open(fullPath);
        if (kernelFile.is_open() && verbose) {
            std::cout << "Loaded kernel from fallback path: " << fullPath << "\n";
        }
    }

    // 3. Bail on failure
    if (!kernelFile.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << "\n";
        status = -1;
        return {};
    }

    // 4. Read kernel source into a string
    std::string source((std::istreambuf_iterator<char>(kernelFile)),
                       std::istreambuf_iterator<char>());
    status = 0;
    return source;
}


int show_norm = -1; // -1 means uninitialized


template<typename T = float, bool static_size = false>
void run_test(
    cl::Context &context, 
    cl::CommandQueue &queue, cl::Program &program,
    const int nq0, const int nq1, const int nq2,
    const int numThreads, 
    const int threadsPerBlockX,
    const int threadsPerBlockY,
    const int threadsPerBlockZ, 
    const int nelmt, const int ntests)
{
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
    const int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

    //Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nm0 * nm1 * nm2, (T)0.0);

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

    const size_t ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2; //shared memory dynamic size
    const size_t sharedMemSize = ssize * sizeof(T);

    cl::Buffer d_basis0(context,basis0.begin(),basis0.end(),true);
    cl::Buffer d_basis1(context,basis1.begin(),basis1.end(),true);
    cl::Buffer d_basis2(context,basis2.begin(),basis2.end(),true);
    cl::Buffer d_JxW(context,JxW.begin(),JxW.end(),true);
    cl::Buffer d_in(context,in.begin(),in.end(),true);
    cl::Buffer d_out(context,out.begin(),out.end(),false);


    // Helper function to select a particular kernel by name
    auto kernel_helper = [&](std::string kernelName) -> cl::Kernel {

        cl::Kernel kernel(program,kernelName);

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
            kernel.setArg(3, nelmt);
            kernel.setArg(4, d_basis0);
            kernel.setArg(5, d_basis1);
            kernel.setArg(6, d_basis2);
            kernel.setArg(7, d_JxW);
            kernel.setArg(8, d_in);
            kernel.setArg(9, d_out);
            kernel.setArg(10, cl::Local(sharedMemSize));
        }

        return kernel;
    };

    // Helper function to calculate performance in GDoF/s
    auto dof_rate = [=](double elapsed) { 
        return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed; 
    };

    auto show_norm_helper = [&]() -> void {
        if (show_norm) {
            // 2. Copy data from device buffer d_out to host
            queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(T) * out.size(), out.data());
            // Evaluate the norm in double precision
            T normSqr = inner_product(out.begin(),out.end(),out.begin(),(T)0);
            std::cout << "# OpenCL kernel norm = " << std::sqrt(normSqr) << '\n';
        }
    };   

    auto print_stats_helper = [&](const cl::Kernel& kernel, double time) {
        std::printf("%s\t%s\t%u\t%f\n",
            static_size ? "static" : "dynamic",
            kernel.getInfo<CL_KERNEL_FUNCTION_NAME>().c_str(),
            nelmt,
            dof_rate(time));
    };

    const size_t globalSize = numBlocks * threadsPerBlock;


    // ------------------------- Kernel with 1D block size -------------------------------
    {
        const std::string kernelName = "BwdTransHexKernel_QP_1D";

        auto kernel = kernel_helper(kernelName);

//    dim3 gridDim(numBlocks)
//    dim3 blockDim(std::min(nq0 * nq1 * nq2, threadsPerBlock))
        cl::NDRange local = cl::NDRange(std::min(nq0 * nq1 * nq2, threadsPerBlock));
        cl::NDRange global = cl::NDRange(globalSize);

        double time = std::numeric_limits<double>::max();
        Timer clTimer;
        for (int t = 0; t < ntests; ++t)
        {
            clTimer.start();
            
            queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,local);
            queue.finish();

            clTimer.stop();
            time = std::min(time, clTimer.elapsedSeconds());
        }

        print_stats_helper(kernel, time);
        show_norm_helper();
    }

    // ------------------------- Kernel with 3D block size -------------------------------
    {
        const std::string kernelName = "BwdTransHexKernel_QP_1D_3D_BLOCKS";

        auto kernel = kernel_helper(kernelName);

//        dim3 gridDim(numBlocks);
//        dim3 blockDim(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));
        cl::NDRange local = cl::NDRange(std::min(nq0,threadsPerBlockX), std::min(nq1,threadsPerBlockY), std::min(nq2, threadsPerBlockZ));
        cl::NDRange global = cl::NDRange(globalSize);

        double time = std::numeric_limits<double>::max();
        Timer clTimer;
        for (int t = 0; t < ntests; ++t)
        {
            clTimer.start();
            
            queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,local);
            queue.finish();

            clTimer.stop();
            time = std::min(time, clTimer.elapsedSeconds());
        }

        print_stats_helper(kernel, time);
        show_norm_helper();
    }

    // ------------------------- Kernel with 3D block size + SimpleMap -------------------------------
    {
        const std::string kernelName = "BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap";

        auto kernel = kernel_helper(kernelName);

//        dim3 gridDim(numBlocks);      // number of blocks in the grid
//        dim3 blockDim(nq0, nq1, nq2); // dimensions of block

        cl::NDRange local = cl::NDRange(nq0, nq1, nq2); // local thread size
        cl::NDRange global = cl::NDRange(numBlocks*nq0,nq1,nq2); // global thread size

        double time = std::numeric_limits<double>::max();
        Timer clTimer;
        for (int t = 0; t < ntests; ++t)
        {
            clTimer.start();
            
            queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,local);
            queue.finish();

            clTimer.stop();
            time = std::min(time, clTimer.elapsedSeconds());
        }

        print_stats_helper(kernel, time);
        show_norm_helper();
    }

}


int main(int argc, char **argv){

    // Default precision
    using real_type = float;

    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s [--help] [nq0 [nq1 [nq2 [nelmt [numThreads [threadsPerBlockX [threadsPerBlockY [threadsPerBlockZ [ntests]]]]]]]]]\n", argv[0]);
        exit(0);
    }

    size_t nq0                = (argc > 1) ? atoi(argv[1]) : 4;
    size_t nq1                = (argc > 2) ? atoi(argv[2]) : 4;
    size_t nq2                = (argc > 3) ? atoi(argv[3]) : 4;
    size_t nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    size_t numThreads         = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    size_t threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    size_t threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    size_t threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    size_t ntests             = (argc > 9) ? atoi(argv[9]) : 50;

    const char *env = getenv("SHOW_NORM");
    show_norm = (env && strcmp(env, "1") == 0) ? 1 : 0;

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
        std::cout << "# Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

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
        std::cout << "# Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        // 5. Create context with the selected device
        cl::Context context(device);

        // 6. Create command queue (enable profiling if needed)
        cl::CommandQueue queue(context, device);


        // ----- Kernels with dynamic size -----
        {
            constexpr bool static_size = false;

            int ierr;
            auto source = loadKernelSource(static_size,ierr);
            if (ierr) return 1;

            std::string buildOptions{"-I ./src "}; // the trailing space is important here

            if constexpr (std::is_same_v<real_type, double>) {
                buildOptions += "-DDOUBLE_PRECISION";
            }

            cl::Program program(context,source);
            try {
                program.build({device},buildOptions);
            } catch (const cl::Error&) {
                // If build fails, print build log
                std::cerr << "Build failed:\n" 
                          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                return 1;
            }

            // Dynamic sizes
            run_test<real_type,static_size>(context, queue, program,           
                nq0, nq1, nq2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);

        }


        // ----- Kernels with static quadrature size -----
        {
            constexpr bool static_size = true;

            int ierr;
            auto source = loadKernelSource(static_size,ierr);
            if (ierr) return 1;

            std::string buildOptions{"-I ./src "}; // the trailing space is important here

            if constexpr (std::is_same_v<real_type, double>) {
                buildOptions += "-DDOUBLE_PRECISION";
            }

            std::unordered_map<std::string, int> defines = {
               {"NQ0", (int) nq0},
               {"NQ1", (int) nq1},
               {"NQ2", (int) nq2}
            };
            
            buildOptions += buildOpenCLDefines(defines);

            cl::Program program(context,source);
            try {
                program.build({device},buildOptions);
            } catch (const cl::Error&) {
                // If build fails, print build log
                std::cerr << "Build failed:\n" 
                          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                return 1;
            }

            run_test<real_type,static_size>(context, queue, program,           
                nq0, nq1, nq2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);

        }

    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }

    return 0;
}