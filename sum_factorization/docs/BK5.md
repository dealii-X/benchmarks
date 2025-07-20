## BK5 Benchmarks and Input Parameters

### 1. **./cuda_benchmark**

This benchmark compares different CUDA kernels implementing the sum factorization algorithm:

- **First kernel**: Uses a 1D grid and 3D block layout, where each block computes each element in strided fashion.
- **Second kernel**: Uses a 1D grid and 3D block layout but simple map approach is used (no-stride).
- **Third kernel**: Uses a 1D grid and 2D block layout and threads are assigned to i and j directions in strided fashion. 
- **Fourth kernel**: Uses a 1D grid and 2D block layout and threads are assigned to j and k directions in strided fashion. 
- **Fifth kernel**: Uses a 1D grid and 2D block layout and threads are also assigned to j and k directions but simple map approach is used (no-stride). 

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 2. **./templated_cuda_benchmark**

This benchmark also compares CUDA kernels but number of quadrature points (nq) are templated.

- **First kernel**: Uses a 1D grid and 3D block layout but simple map approach is used (no-stride).
- **Second kernel**: Uses a 1D grid and 2D block layout and threads are also assigned to j and k directions but simple map approach is used (no-stride).

**Input parameters:**
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 3. **./kokkos_benchmark**

This benchmark also compares CUDA kernels but number of quadrature points (nq) are templated.

- **First kernel**: Uses a 1D grid and 3D block layout but simple map approach is used (no-stride).
- **Second kernel**: Uses a 1D grid and 2D block layout and threads are also assigned to j and k directions but simple map approach is used (no-stride).

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlock**: Number of threads per block
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 4. **./templated_kokkos_benchmark**

This benchmark also compares Kokkos kernels but number of quadrature points (nq) are templated.

- **First kernel**: Uses a 1D grid and 3D block layout but simple map approach is used (no-stride).
- **Second kernel**: Uses a 1D grid and 2D block layout and threads are also assigned to j and k directions but simple map approach is used (no-stride).

**Input parameters:**
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)



## GPU profiling (CUDA only)
To gain better insight and obtain detailed performance metrics, reports from **Nsight Systems** and **Nsight Compute** can be automatically generated — provided that the appropriate `nsys` and `ncu` environment variables are set. Run the following to generate reports:

```bash
make reportBK5
```
The reports will be saved in the **`<build_dir>/BK5/reports/`** directory.

To view the results using the Nvidia profiler tools:
```bash
cd build/BK5/reports
ncu-ui benchmark_3_3_3_400000_27000_1.ncu-rep
nsys-ui benchmark_3_3_3_400000_27000_1.nsys-rep

```