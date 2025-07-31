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


std::string loadKernelSource(const std::string &filename, int& status, bool verbose = false) {

    //std::string filename = static_size ? "opencl_kernels_static.cl" : "opencl_kernels.cl";
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

template<typename T = float, int nq0, int nq1, int nq2>
void run_test(
    cl::Context &context,
    cl::CommandQueue &queue, cl::Program &program,
    const unsigned int nelmt, const int ntests)
{
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

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

    cl::Buffer d_basis0(context,basis0.begin(),basis0.end(),true);
    cl::Buffer d_basis1(context,basis1.begin(),basis1.end(),true);
    cl::Buffer d_basis2(context,basis2.begin(),basis2.end(),true);
    cl::Buffer d_JxW(context,JxW.begin(),JxW.end(),true);
    cl::Buffer d_in(context,in.begin(),in.end(),true);
    cl::Buffer d_out(context,out.begin(),out.end(),false);

    // Helper function to select a particular kernel by name
    auto kernel = [&]() -> cl::Kernel {

        cl::Kernel kernel(program,"sum_factorization");

        kernel.setArg(0, d_basis0);
        kernel.setArg(1, d_basis1);
        kernel.setArg(2, d_basis2);
        kernel.setArg(3, d_JxW);
        kernel.setArg(4, JxW.size());
        kernel.setArg(5, d_in);
        kernel.setArg(6, in.size());
        kernel.setArg(7, d_out);
        kernel.setArg(8, out.size());

        return kernel;
    }();

    const size_t ndofs = nelmt * nm0 * nm1 * nm1;

    // Helper function to calculate performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed;
    };

    auto show_norm_helper = [&]() -> void {
        if (show_norm) {
            // 2. Copy data from device buffer d_out to host
            queue.enqueueReadBuffer(d_out, CL_TRUE, 0, sizeof(T) * out.size(), out.data());

            double normSqr = 0.0;
            for (auto d : out) {
                double dd = d;
                normSqr += dd * dd;
            }

            std::cout << "# OpenCL kernel norm = " << std::sqrt(normSqr) << '\n';
        }
    };

    auto print_stats_helper = [&](const cl::Kernel& kernel, double time) {
        std::printf("%s\t%u\t%f\n",
            kernel.getInfo<CL_KERNEL_FUNCTION_NAME>().c_str(),
            nelmt,
            dof_rate(time));
    };

    {
        cl::NDRange local = cl::NullRange;
        cl::NDRange global = cl::NDRange(64,1,nelmt);

        double time = std::numeric_limits<double>::max();
        Timer clTimer;
        for (int t = 0; t < ntests; ++t)
        {
            clTimer.start();

            queue.enqueueNDRangeKernel(kernel,cl::NullRange,global,cl::NullRange);
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
        printf("Usage: %s [--help] [nelmt [ntests]]\n", argv[0]);
        exit(0);
    }

    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    unsigned int ntests             = (argc > 2) ? atoi(argv[2]) : 5u;

    const char *env = getenv("SHOW_NORM");
    show_norm = (env && strcmp(env, "1") == 0) ? 1 : 0;

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


        // --- TinyTC Kernel ---
        {
            int ierr;
            auto source = loadKernelSource("sum_factorization.cl",ierr);
            std::string buildOptions{""};

            cl::Program program(context,source);
            try {
                program.build({device},buildOptions);
            } catch (const cl::Error &) {
                std::cerr << "Build failed:\n" 
                          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
                return 1;
            }

            run_test<float,4,4,4>(context,queue,program,nelmt,ntests);
        }


    } catch (cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    }

    return 0;
}
