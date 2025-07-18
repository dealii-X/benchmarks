This folder implements the CEED Bakeoff problems using alternative programming models.
The main purpose is to allow evaluation across a wider array of accelerator devices.

Currently, only the following two models are included:

- OpenCL
- SYCL

The kernels have been ported manually (i.e. search & replace) from the CUDA variants in the [sum_factorization/][../sum_factorization/] folder.

## Prerequisites

- OpenCL
- [OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP)

## Setup

### Environment variables

- `SHOW_NORM=<0,1>`: show square norm of output array (only for quick validation)
- `CL_KERNEL_DIR=<path>`: folder containing the kernels (still subject of change...)


## Resources

- [OpenCL](https://www.khronos.org/api/index_2017/opencl)
- [OpenCL-Guide](https://github.com/KhronosGroup/OpenCL-Guide)
- [OpenCL C++ Bindings (doxygen)](https://github.khronos.org/OpenCL-CLHPP/)
- [OpenCL™ Programming Guide for the CUDA™ Architecture](https://developer.download.nvidia.com/compute/DevZone/docs/html/OpenCL/doc/OpenCL_Programming_Guide.pdf) (PDF, 1503 KB)
- [StackOverflow [opencl]](https://stackoverflow.com/questions/tagged/opencl)
- [r/OpenCL](https://www.reddit.com/r/OpenCL/)
- [Intel® Tools for OpenCL™ Software](https://www.intel.com/content/www/us/en/developer/tools/opencl/overview.html)
- [OpenCL Programming Guide for Mac](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/OpenCL_MacProgGuide/Introduction/Introduction.html)
- [Hands On OpenCL](https://handsonopencl.github.io/)
- [OpenCL on Linux](https://github.com/bashbaug/OpenCLPapers/blob/master/OpenCLOnLinux.asciidoc)
- [oneAPI GPU Optimization Guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/overview.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Tools

- [clinfo](https://github.com/Oblomov/clinfo)
- [opencl-intercept-layer](https://github.com/intel/opencl-intercept-layer)
- [opencl-kernel-profiler](https://github.com/rjodinchr/opencl-kernel-profiler)
- [OpenCL-Wrapper](https://github.com/ProjectPhysX/OpenCL-Wrapper)
- [opencl-extension-loader](https://github.com/bashbaug/opencl-extension-loader)

## Other OpenCL projects

- [SimpleOpenCLSamples](https://github.com/bashbaug/SimpleOpenCLSamples)
- [OpenCL-Benchmark](https://github.com/ProjectPhysX/OpenCL-Benchmark)
