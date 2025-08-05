# CEED Benchmarks - Alternatives

This folder implements the *CEED Benchmarks* using alternative programming models.

The purpose is to evaluate the benchmarks across a wider array of accelerator devices in addition to the CUDA and Kokkos variants found in the main [`sum_factorization/`](../sum_factorization/) folder.

The alternative models considered include:
- OpenMP
- OpenCL

The kernels have been ported manually (i.e. search & replace) from the CUDA variants in the original [`sum_factorization/`](../sum_factorization/) folder. 

Drivers using SYCL and [TinyTC](https://github.com/intel/tiny-tensor-compiler) are a work in progress.

## Dependencies

For OpenCL-based benchmarks (including the TinyTC variants):
- OpenCL (Apple, Intel, Nvidia, ...)
- [OpenCL-CLHPP](https://github.com/KhronosGroup/OpenCL-CLHPP)

For TinyTC-based benchmarks:
- [Tiny Tensor Compiler](https://github.com/intel/tiny-tensor-compiler)

For SYCL benchmarks, select one:
- [Intel oneAPI DPC++/C++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp)

## Setup

For the default OpenMP variants:

```
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=<compiler-of-choice>
make -j4
```

### Build options


| Option       | Description           | Default value |
|--------------|-----------------------|---------------|
| BUILD_OPENCL | Build OpenCL variants | OFF           |
| BUILD_SYCL   | Build SYCL variants   | OFF           |
| BUILD_TINYTC | Build TinyTC variants | OFF           |


The OpenCL variants expect the following environment variable is set:

```
set OPENCL_CLHPP_ROOT=<path/to/OpenCL-CLHPP>
```

Note that the OpenMP drivers are always available as they double also as serial validation.
To run the serial version set the environment variable `OMP_TARGET_OFFLOAD=DISABLED`.

### Environment variables

Some settings are configurable for ease of use:

- `SHOW_NORM=0|1`: show square norm of output array (only for quick validation)
- `CL_KERNEL_DIR=<path>`: folder containing the OpenCL kernels (still subject of change...)


## Future ideas

- Translation from CUDA using [Intel DPC++ Compatibility Tool](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compatibility-tool.html)
- Translation using [Coccinelle](https://coccinelle.gitlabpages.inria.fr/website/)
- JIT-Compilation:
  - [Jitify2 (Nvidia)](https://github.com/NVIDIA/jitify/tree/jitify2)
  - [Jitify (ROCm)](https://github.com/ROCm/jitify)

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
