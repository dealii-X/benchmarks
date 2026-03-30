## BK1 Benchmarks and Input Parameters
### 1. **templated_kokkos_benchmark**

This benchmark compares the performance of the sum factorization algorithm implemented in **Kokkos**.
 
**Input parameters:**
- **p**: Polynomial order of each element in one direction 
- **nelmt**: Number of elements  
- **numBlocks**: Number of total threads 
- **threadsPerBlock**: Number of threads per block  
- **ntests**: Number of benchmark repetitions (Best across all tests, used as a reference for comparison)

### 2. **templated_cuda_benchmark**

This benchmark compares the performance of the sum factorization algorithm implemented in **Cuda**.
 
**Input parameters:**
- **p**: Polynomial order of each element in one direction 
- **nelmt**: Number of elements  
- **numBlocks**: Number of total threads 
- **threadsPerBlock**: Number of threads per block  
- **ntests**: Number of benchmark repetitions (Best across all tests, used as a reference for comparison)


### 3. **templated_batched_kokkos_benchmark**

This benchmark is designed for plotting and evaluates multiple polynomial orders with varying total degrees of freedom. No input parameter is required.

### 4. **serial_benchmark**

A serial implementation of the same algorithm is used to verify the parallel implementations.




