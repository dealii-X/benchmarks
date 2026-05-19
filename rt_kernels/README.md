# Raviart-Thomas Kernels

This repository implements **H(div)-conforming** vector field FEM discretizations using **Raviart–Thomas** elements on GPUs through the Kokkos portability layer. Hexahedral elements used and evaluated in a matrix-free formulation.

## Key Implementations
The repository provides high-performance GPU implementations for each individual operator:

1. **Mass Operator** (`u` → `u`)
2. **Mixed Gradient Operator** (`p` → `u`)
3. **Mixed Divergence Operator** (`u` → `p`)

<br>
These kernels can be used to solve a Darcy-flow-like mixed system of equations structured as follows:

| Column: `u` | Column: `p` | RHS |
| :--- | :--- | :--- |
| **Mass Matrix** (`u → u`) | **Mixed Gradient** (`p → u`) | `f` |
| **Mixed Divergence** (`u → p`) | `0` | `g` |

---


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




