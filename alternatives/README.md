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
  - [SYCL Specialization Constants](https://github.khronos.org/SYCL_Reference/iface/specialization-constants.html)

## Other Notes

### TinyTC installation 

The TinyTC-based driver currently targets the `develop` branch of TinyTC. To install the develop version create a local Spack repo and override the Tiny Tensor Compiler package using:

```python
from spack.pkg.builtin.tiny_tensor_compiler import TinyTensorCompiler as BuiltinTinyTC

class TinyTensorCompiler(BuiltinTinyTC):
    git = "https://github.com/intel/tiny-tensor-compiler.git"
    version("develop", branch="develop")
```

To install the develop version use,

```
spack install tiny-tensor-compiler@develop%oneapi@+sycl ^oneapi-level-zero%oneapi
```

Until the CMake build is completed, the following paths can be used to build the TinyTC-based driver:

```
export TINYTC_ROOT=`spack location -i tiny-tensor-compiler`
export LEVELZERO_ROOT=`spack location -i oneapi-level-zero`
```

In case multiple variants are installed use `spack find -lv <package>` to narrow down the one you want to use.
