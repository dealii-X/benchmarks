# CEED Bake-Off Problem 3 (BP3) Benchmark
This benchmark suite implements the [CEED benchmark](https://ceed.exascaleproject.org/bps/) Bake-Off Problem 3 using the deal.II finite element library. Performance portability is achieved by combining Kokkos for shared-memory parallelism with GPU-aware MPI for efficient multi-GPU communication.

## Installation and Build Requirements
The build process follows a specific dependency chain:

1. Kokkos must be compiled for the target GPU architecture (e.g. CUDA or HIP).

2. deal.II must be configured with support for Kokkos, p4est and MPI.

```bash
# NVIDIA
export OMPI_CXX=$(which nvcc_wrapper) 

# AMD
export OMPI_CXX=$(which hipcc)

# Intel
export OMPI_CXX=$(which icpx)

mkdir build && cd build
cmake .. 
-DKokkos_DIR=<KokkosConfig_cmake_dir>  \
-Ddeal.II_DIR=<dealii_Config_cmake_dir> \
-DCMAKE_CXX_COMPILER=mpicxx 

make
```

## Running the Benchmark
The benchmark requires a 1-process-per-GPU configuration. The user must handle device-to-process assignments manually by slurm-specific binding options (i.e srun --gpus-per-task=1).

```bash
mpirun -n <num_process> ./build/bp3 <poly_degree> <min_dofs> <max_dofs>
```