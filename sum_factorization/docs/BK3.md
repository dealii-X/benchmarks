## BK3 Benchmarks and Input Parameters

### 1. **./cuda_benchmark**

This benchmark compares different CUDA kernels implementing the sum factorization algorithm:

- **First kernel**: Uses a 1D grid and 3D block layout, where each block computes each element in strided fashion.
- **Second kernel**: Uses a 1D grid and 3D block layout but simple map approach is used (no-stride).
- **Third kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions in strided fashion. 
- **Fourth kernel**: Uses a 1D grid and 2D block layout and threads are assigned to p and q directions but simple map approach is used (no-stride). 

**Input parameters:**
- **nq0**, **nq1**, **nq2**: Quadrature points in each dimension (element dof per direction + 1)
- **nelmt**: Number of elements
- **numThreads3D**: Number of total threads for 3D block kernels
- **threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ**: Number of threads per block in each dimension
- **ntests**: Number of benchmark repetitions (minimum across all tests, used as a reference for comparison)

