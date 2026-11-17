# Elasticity Kernels

This repository implements **$H^1$-conforming** 3D vector field FEM discretizations for **elasticity** on GPUs through the **Kokkos** portability layer. Hexahedral elements are used and evaluated in a matrix-free sum-factorized formulation.

## Key Implementations
The repository provides high-performance implementations for the isotropic **Linear Elasticity Stiffness Operator**.

1. **Serial Sum-Factorization Kernel** (`LinearElasticity::Serial::SumFactorization`) — CPU reference implementation
2. **Kokkos Parallel Kernel** (`LinearElasticity::Parallel::Kokkos_LinearElasticityOperator`) — GPU team-based implementation using shared-memory scratch space and register blocking

<br>
These kernels evaluate the weak form `a(u, v) = ∫ ε(v) : C : ε(u) dΩ` of the Navier–Lamé linear elasticity equations, coupling all three displacement components `(u_x, u_y, u_z)` via `ε(u) = ½(∇u + ∇uᵀ)` and `σ(u) = λ tr(ε(u)) I + 2μ ε(u)`:



## Software Dependencies

- [Kokkos](https://github.com/kokkos/kokkos)

## Building the Project

### Kokkos with CUDA backend

```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=<KokkosConfig_cmake_dir> -DCMAKE_CXX_COMPILER=nvcc_wrapper 
make -j
```

### Kokkos with HIP backend

```bash
mkdir build && cd build
cmake .. -DKokkos_DIR=<KokkosConfig_cmake_dir> -DCMAKE_CXX_COMPILER=hipcc
make -j
```

### Kokkos with SYCL backend

```bash
source /opt/intel/oneapi/setvars.sh
mkdir build && cd build
cmake .. -DKokkos_DIR=<KokkosConfig_cmake_dir> -DCMAKE_CXX_COMPILER=icpx 
make -j
```