# Portable Halo Exchange Microbenchmark (p-halox)

**p-halox** microbenchmark is designed to measure communication bottlenecks in
MPI-based local neighbor exchange (halo exchange) algorithms.

To target multiple architectures (CPU and GPU), **Kokkos** is used for portability.

## Software Dependencies
- MPI with device support (GPU-aware MPI)
- [Kokkos](https://github.com/kokkos/kokkos) (built with the appropriate backend enabled)

## Build Configurations
### CUDA

```bash
export OMPI_CXX=nvcc_wrapper

mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DKokkos_DIR=<KokkosConfig_cmake_dir>
make
```

### HIP (ROCm)
```bash
export OMPI_CXX=hipcc

mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DKokkos_DIR=<KokkosConfig_cmake_dir>
make
```

### SYCL (ONEAPI)
```bash
source /opt/intel/oneapi/setvars.sh
export OMPI_CXX=icpx

mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=mpicxx -DKokkos_DIR=<KokkosConfig_cmake_dir>
make
```

## Usage
### Input Parameters
- **dim** = Cartesian grid dimension (1, 2, or 3)
- **KB**  = Message size per direction (in KB)
- **nMsg**= Number of messages transferred per direction per neighbor
- **is_periodic** = Set to 1 for periodic boundaries, 0 otherwise
- **warmup** = Number of warm-up iterations before timing starts
- **print_topo** = Set to 1 to print topology information for the neighborhood relation, 0 otherwise

### Running the Application
mpirun -np 4 ./p-halox 2 128 10 1 5 0

**Note:** p-halox requires one GPU per MPI process (e.g. `srun --gpus-per-task=1`).