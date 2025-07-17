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
[Bake-off Kernel 1](docs/BK1.md) : scalar E-vector evaluation of mass matrix, q = p + 2 (Gauss-Legendre quadrature). <br>
[Bake-off Kernel 5](docs/BK5.md) : scalar E-vector-to-E-vector evaluation of stiffness matrix, q=p+1 (Gauss-Lobatto-Legendre).
