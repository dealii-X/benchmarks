// bk5_opencl.cc

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120  // or 120, 100, depending on what you want
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>  // or cl2.hpp if you prefer

#include <iostream>
#include <cstdio>
#include <fstream>
#include <execution>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <numeric>

#include "timer.hpp"
#include "common.hpp"

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

    std::string filename = static_size ? "BK5_static.cl" : "BK5.cl";
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



bool show_norm = false;

template<typename T, bool static_size = false>
void run_test(
    cl::Context &context, 
    cl::CommandQueue &queue, cl::Program &program,
    const int nq0, const int nq1, const int nq2,
    const int numThreads, 
    const int threadsPerBlockX, 
    const int threadsPerBlockY, 
    const int threadsPerBlockZ,
    const int nelmt, const int ntests,
    const int maxWorkGroupSize)
{

    // Initialize the input and output arrays
    std::vector<T> dbasis0(nq0 * nq0);
    std::vector<T> dbasis1(nq1 * nq1);
    std::vector<T> dbasis2(nq2 * nq2);

    std::vector<T> G(nelmt * nq0 * nq1 * nq2 * 6, (T)2.0);
    std::vector<T> in(nelmt * nq0 * nq1 * nq2, (T)3.0);
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, (T)0.0);

    //Initialization of basis functions
    for(int i = 0u; i < nq0; i++) {
        for(int a = 0u; a < nq0; a++) {
            dbasis0[i * nq0 + a] = std::cos((T)(i * nq0 + a));
        }
    }

    for(int j = 0u; j < nq1; j++) {
        for(int b = 0u; b < nq1; b++) {
            dbasis1[j * nq1 + b] = std::cos((T)(j * nq1 + b));
        }
    }

    for(int k = 0u; k < nq2; k++) {
        for(int c = 0u; c < nq2; c++) {
            dbasis2[k * nq2 + c] = std::cos((T)(k * nq2 + c));
        }
    }

    cl::Buffer d_dbasis0(context,dbasis0.begin(),dbasis0.end(),true);
    cl::Buffer d_dbasis1(context,dbasis1.begin(),dbasis1.end(),true);
    cl::Buffer d_dbasis2(context,dbasis2.begin(),dbasis2.end(),true);
    cl::Buffer d_G(context,G.begin(),G.end(),true);
    cl::Buffer d_in(context,in.begin(),in.end(),true);
    cl::Buffer d_out(context,out.begin(),out.end(),false);

    // Helper function to select a particular kernel by name
    auto kernel_helper = [&](std::string kernelName, size_t sharedMemSize) -> cl::Kernel {

        cl::Kernel kernel(program,kernelName);

        kernel.setArg(0, nq0);
        kernel.setArg(1, nq1);
        kernel.setArg(2, nq2);
        kernel.setArg(3, nelmt);
        kernel.setArg(4, d_dbasis0);
        kernel.setArg(5, d_dbasis1);
        kernel.setArg(6, d_dbasis2);
        kernel.setArg(7, d_G);
        kernel.setArg(8, d_in);
        kernel.setArg(9, d_out);
        kernel.setArg(10, cl::Local(sharedMemSize));

        return kernel;
    };


    // Helper function to calculate performance in GDoF/s
    auto dof_rate = [=](double elapsed) { 
        return 1.0e-9 * nelmt * nq0 * nq1 * nq2 / elapsed; 
    };

    auto print_stats_helper = [&](const cl::Kernel& kernel, double time) {
        
        std::printf("%s\t%s\t%u\t%f\n",
            static_size ? "static" : "dynamic",
            kernel.getInfo<CL_KERNEL_FUNCTION_NAME>().c_str(),
            nelmt,
            dof_rate(time));

        if (show_norm) {
            // 2. Copy data from device buffer d_out to host
            queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(T) * out.size(), out.data());
            // Evaluate the norm in double precision
            T normSqr = squared_norm<T,double>(out.data(),out.size());
            std::cout << "# OpenCL kernel norm = " << std::sqrt(normSqr) << '\n';
        }
    };

    auto run_kernel = [&](cl::Kernel& kernel, cl::NDRange& global, cl::NDRange&local) {
        double time = std::numeric_limits<double>::max();
        Timer clTimer;
        for (int t = 0u; t < ntests; ++t)
        {   
            clTimer.start();

            queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,local);
            queue.finish();

            clTimer.stop();
            time = std::min(time, clTimer.elapsedSeconds());
        }
        return time;
    };

//    const size_t globalSize = numBlocks * threadsPerBlock;
    const size_t globalSize = numThreads;


//    int device;   cudaGetDevice(&device);   cudaDeviceProp prop;
//    cudaGetDeviceProperties(&prop, device);
//        prop.maxThreadsPerBlock
//    const size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // ------------------------- Kernel with 3D block size + Simple Map -------------------------------
    if(nq0 * nq1 * nq2 < maxWorkGroupSize)
    {   
        const int numBlocks = numThreads / (nq0 * nq1 * nq2);
        int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;
        
        auto kernel = kernel_helper("TransHexKernel_QP_3D_Block_SimpleMap",static_cast<size_t>(ssize));

        cl::NDRange global = cl::NDRange(globalSize);
        cl::NDRange local = cl::NDRange(nq0 * nq1 * nq2);

        double time = run_kernel(kernel,global,local);
        print_stats_helper(kernel, time);
    }


    // ------------------------- Kernel with 3D block size -------------------------------
    {   
        int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
        const int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));
        int ssize = nq0 * nq0 + nq1 * nq1 + nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size

        auto kernel = kernel_helper("TransHexKernel_QP_3D_Block",static_cast<size_t>(ssize));

        cl::NDRange global = cl::NDRange(globalSize);
        cl::NDRange local = cl::NDRange(std::min(nq0 * nq1 * nq2, threadsPerBlock));

        double time = run_kernel(kernel,global,local);
        print_stats_helper(kernel, time);
    }


    // ------------------------- Kernel with 2D block size (ij)-------------------------------
    {   
        int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        const int numBlocks = numThreads / (std::min(nq0 * nq1, threadsPerBlock));
        int ssize = nq2 * nq2 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        auto kernel = kernel_helper("TransHexKernel_QP_2D_Block_ij",static_cast<size_t>(ssize));

// dim3 gridDim(numBlocks)
// dim3 blockDim(std::min(nq0 * nq1, threadsPerBlock))
        cl::NDRange global = cl::NDRange(globalSize);
        cl::NDRange local = cl::NDRange(std::min(nq0 * nq1, threadsPerBlock));

        double time = run_kernel(kernel,global,local);
        print_stats_helper(kernel, time);
    }


    // ------------------------- Kernel with 2D block size (jk)-------------------------------
    {   
        int threadsPerBlock = threadsPerBlockX * threadsPerBlockY;
        const int numBlocks = numThreads / (std::min(nq1 * nq2, threadsPerBlock));
        int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        auto kernel = kernel_helper("TransHexKernel_QP_2D_Block_jk",static_cast<size_t>(ssize));

// dim3 gridDim(numBlocks)
// dim3 blockDim(std::min(nq1 * nq2, threadsPerBlock))
        cl::NDRange global = cl::NDRange(globalSize);
        cl::NDRange local = cl::NDRange(std::min(nq0 * nq1, threadsPerBlock));

        double time = run_kernel(kernel,global,local);
        print_stats_helper(kernel, time);
    }


    // ------------------------- Kernel with 2D block size (jk) Simple Map-------------------------------
    if(nq1 * nq2 < maxWorkGroupSize)
    {   
        const int numBlocks = numThreads / (nq1 * nq2);
        int ssize = nq0 * nq0 + 3 * nq0 * nq1 * nq2;          //shared memory dynamic size
    
        auto kernel = kernel_helper("TransHexKernel_QP_2D_Block_jk_SimpleMap",static_cast<size_t>(ssize));

// dim3 gridDim(numBlocks)
// dim3 blockDim(nq1 * nq2)
        cl::NDRange global = cl::NDRange(globalSize);
        cl::NDRange local = cl::NDRange(nq1 * nq2);

        double time = run_kernel(kernel,global,local);
        print_stats_helper(kernel, time);
    }

}



// Select platform and device, with optional fallback to CPU
cl::Device selectDevice(bool preferGPU = true) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
        throw std::runtime_error("No OpenCL platforms found.");

    cl::Device selectedDevice;
    for (auto& platform : platforms) {
        std::cout << "# Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";

        std::vector<cl::Device> devices;
        if (preferGPU) {
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (devices.empty()) {
                std::cout << "No GPU devices found on this platform; trying CPU devices...\n";
                platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            }
        } else {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        }

        if (!devices.empty()) {
            selectedDevice = devices[0];
            std::cout << "# Selected device: " << selectedDevice.getInfo<CL_DEVICE_NAME>() << "\n";
            return selectedDevice;
        }
    }
    throw std::runtime_error("No suitable OpenCL devices found.");
}

// Create OpenCL context and queue for a device
std::pair<cl::Context, cl::CommandQueue> createContextAndQueue(const cl::Device& device, bool enableProfiling = false) {
    cl::Context context(device);
    cl_command_queue_properties props = enableProfiling ? CL_QUEUE_PROFILING_ENABLE : 0;
    cl::CommandQueue queue(context, device, props);
    return {context, queue};
}

// Build OpenCL program with error reporting
cl::Program buildProgram(const cl::Context& context, const cl::Device& device, const std::string& source,
                        const std::string& buildOptions = "") {
    cl::Program program(context, source);
    try {
        program.build({device}, buildOptions);
    } catch (const cl::Error&) {
        std::cerr << "OpenCL Program build failed:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        throw;
    }
    return program;
}

int main(int argc, char **argv){

    // Default precision
    using real_type = float;


    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s [--help] [nq0 [nq1 [nq2 [nelmt [numThreads [threadsPerBlockX [threadsPerBlockY [threadsPerBlockZ [ntests]]]]]]]]]\n", argv[0]);
        exit(0);
    }

    int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    int numThreads         = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    int threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    int threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    int threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    int ntests             = (argc > 9) ? atoi(argv[9]) : 50u;

    const char *env = std::getenv("SHOW_NORM");
    show_norm = (env && strcmp(env, "1") == 0);

    std::cout.precision(8);

    try {
        cl::Device device = selectDevice(/*preferGPU=*/true);
        auto [context, queue] = createContextAndQueue(device, /*enableProfiling=*/false);

        const size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

        // ----- Kernels with dynamic size -----
        {
            constexpr bool static_size = false;

            int ierr;
            auto source = loadKernelSource(static_size, ierr);
            if (ierr) {
                std::cerr << "Failed to load kernel source.\n";
                return 1;
            }

            std::string buildOptions{"-I ./src "};
            if constexpr (std::is_same_v<real_type, double>) {
                buildOptions += "-DDOUBLE_PRECISION";
            }

            cl::Program program = buildProgram(context, device, source, buildOptions);

            run_test<real_type, static_size>(context, queue, program,
                nq0, nq1, nq2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests,
                maxWorkGroupSize);            
        }

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}