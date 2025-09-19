# CEED Benchmarks
This project demonstrates the implementation and performance comparison of [high order matrix-free FEM](docs/presentations/) based on the [CEED benchmarks](https://ceed.exascaleproject.org/bps/). Implementations are based on the **CUDA** and **Kokkos** programming models and targets single GPU. For algorithm verification and comparison purposes, serial versions of the kernels are also included.

## Software Dependencies

- [Kokkos](https://github.com/kokkos/kokkos)

## Building the Project

### Kokkos with CUDA backend

```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=<KokkosConfig_cmake_dir> -DCMAKE_CXX_COMPILER=nvcc_wrapper 
make -j
```

### Kokkos with SYCL backend

```bash
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DKokkos_DIR=<KokkosConfig_cmake_dir> -DCMAKE_CXX_COMPILER=icpx 
make -j
```
## Implementations
[Bake-off Kernel 1](docs/BK1.md) : scalar E-vector-to-E-vector evaluation of mass matrix, q = p + 2 (Gauss-Legendre). <br>
[Bake-off Kernel 3](docs/BK3.md) : scalar E-vector-to-E-vector evaluation of stifness matrix, q = p + 1 (Gauss-Legendre). <br>
[Bake-off Kernel 5](docs/BK5.md) : scalar E-vector-to-E-vector evaluation of stiffness matrix, q = p + 1 (Gauss-Lobatto-Legendre).

## Post-processing
### 1. Generating DOF-GDOF/s Plots and Kernel Outputs
To gain deeper insights and more effectively analyze the results, DOF-GDOF/s plots and kernel outputs can be quickly generated using the provided Python script. To see this in action, the following illustrates an example use case:

```bash
cd <build_dir>/outputs
../BK3/cuda_benchmark 4 4 4 500000 4000000 4 4 4 10 >> BK3_cuda.txt
../BK3/cuda_benchmark 8 8 8 100000 4000000 8 8 8 10 >> BK3_cuda.txt
python3 plot.py BK3_cuda.txt
```

### 2. Roofline Model
Provided python script can generate a roofline model and print the achieved kernel performance values. Example usage of the Python script:

```bash
cd <build_dir>/roofline
# Modifying the "Input Parameters" in rooflineBK?.py
python3 rooflineBK?.py
```
The theoretical bandwidth and TFLOPs of NVIDIA devices can be calculated using simple formulas.

$$
\boxed{\text{Global Memory Bandwidth (GB/s)} = \text{Memory Clock (GHz)} \times \text{Bus Width (bits)} / 8}
$$

$$
\boxed{\text{Shared Memory Bandwidth (GB/s)} = \text{SM Clock (GHz)} \times \text{\#SM} \times \text{word length} \times \text{\#Banks}}
$$

