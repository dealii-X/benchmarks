# Sum Factorization Kernels
This project demonstrates the implementation and performance comparison of sum factorization kernels using both **CUDA** and **Kokkos**. For algorithm verification and comparison purposes, serial versions of the kernels are also included.

## Features

- CUDA and Kokkos-based GPU implementations
- Serial baseline for verification
- Performance benchmarking

## Benchmarks and Input Parameters
### 1. **./benchmark**

This benchmark compares the performance of the sum factorization algorithm implemented in **CUDA** and **Kokkos**.

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)  
- **numThreads**: Number of total threads 
- **threadsPerBlock**: Number of threads per block  
- **nelmt**: Number of elements  
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 2. **./block_dimension_benchmark**

This benchmark compares three different CUDA kernels implementing the same sum factorization algorithm:

- **First kernel**: Identical to the original **benchmark** kernel, using a 1D grid and block configuration.
- **Second kernel**: Uses a 3D grid and block layout.
- **Third kernel**: Also employs a 3D grid and block, but with a simplified thread-to-data mapping — each thread processes one data element.

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)  
- **numThreadsX, numThreadsY, numThreadsZ**: Number of total threads in each dimension
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **nelmt**: Number of elements  
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)


## Software Dependencies

- [Kokkos](https://github.com/kokkos/kokkos) (must be installed with GPU backend enabled)

## Building the Project

After cloning the repository:

```bash
mkdir build && cd build
cmake -DENABLE_TESTS=OFF ..
make
```

## GPU profiling (CUDA only)

To gain better insight and obtain detailed performance metrics, reports from **Nsight Systems** and **Nsight Compute** can be automatically generated — provided that the appropriate `nsys` and `ncu` environment variables are set. Run the following to generate reports:

```bash
make report
```
The reports will be saved in the build/reports/ directory.

To view the results using the Nvidia profiler tools:
```bash
cd build/reports
ncu-ui benchmark_4_4_4_256_16_2000_1.ncu-rep
nsys-ui benchmark_4_4_4_256_16_2000_1.nsys-rep

```