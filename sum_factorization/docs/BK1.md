## BK1 Benchmarks and Input Parameters
### 1. **./kokkos_benchmark**

This benchmark compares the performance of the sum factorization algorithm implemented in **Kokkos**.

- **First kernel**: Uses a 1D grid and 3D block layout, but cuda block is defined in 1D which contains all the threads.
- **Second kernel**: Uses a 1D grid and 3D block layout, but cuda block is defined in 1D which contains all the threads. Simple map approach is used (no-stride).
- **Third kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions in strided fashion. 
- **Fourth kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions but simple map approach is used (no-stride). 

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)  
- **nelmt**: Number of elements  
- **numThreads**: Number of total threads 
- **threadsPerBlock**: Number of threads per block  
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 2. **./templated_kokkos_benchmark**

This benchmark also compares same Kokkos kernels but number of quadrature points (nq) are templated.

**Input parameters:**
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 3. **./cuda_benchmark**

This benchmark compares different CUDA kernels implementing the sum factorization algorithm:

- **First kernel**: Uses a warp-centric approach, where each warp computes each element.
- **Second kernel**: Also uses a warp-centric approach but only works for linear elements (Q1, nm=2).  
- **Third kernel**: Uses a 1D grid and 3D block layout, but cuda block is defined in 1D which contains all the threads.
- **Fourth kernel**: Uses a 1D grid and 3D block layout, but cuda block is defined in 1D which contains all the threads. Simple map approach is used (no-stride).
- **Fifth kernel**: Uses a 1D grid and 3D block layout.
- **Sixth kernel**: Uses a 1D grid and 3D block layout, but simple map approach is used (no-stride).
- **Seventh kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions in strided fashion. 
- **Eighth kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions but simple map approach is used (no-stride). 

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

### 4. **./templated_cuda_benchmark**

This benchmark also compares same CUDA kernels but number of quadrature points (nq) are templated.

**Input parameters:**
- **nelmt**: Number of elements
- **numThreads**: Number of total threads
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)


## GPU profiling (CUDA only)
To gain better insight and obtain detailed performance metrics, reports from **Nsight Systems** and **Nsight Compute** can be automatically generated — provided that the appropriate `nsys` and `ncu` environment variables are set. Run the following to generate reports:

```bash
make reportBK1
```
The reports will be saved in the **`<build_dir>/BK1/reports/`** directory.

To view the results using the Nvidia profiler tools:
```bash
cd build/BK1/reports
ncu-ui benchmark_3_3_3_400000_27000_1.ncu-rep
nsys-ui benchmark_3_3_3_400000_27000_1.nsys-rep