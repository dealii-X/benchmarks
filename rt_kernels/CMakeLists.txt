cmake_minimum_required(VERSION 3.20)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/cuda_kernels)
include_directories(${CMAKE_SOURCE_DIR}/include/kokkos_kernels)
include_directories(${CMAKE_SOURCE_DIR}/include/serial_kernels)


set(KOKKOS_BENCHMARKS kokkosMassBenchmark kokkosMixedDivBenchmark kokkosMixedGradBenchmark)
set(CUDA_BENCHMARKS mma_f64_m8n8k4_mass_benchmark mma_tf32_m16n8k4_mass_benchmark)

#Kokkos Benchmarks
foreach(benchmark IN LISTS KOKKOS_BENCHMARKS)
    add_executable("${benchmark}" "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}" Kokkos::kokkos)
endforeach()

#Cuda Benchmarks
if(Kokkos_ENABLE_CUDA)
    foreach(benchmark IN LISTS CUDA_BENCHMARKS)
        add_executable("${benchmark}" "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/${benchmark}.cc")
        TARGET_LINK_LIBRARIES("${benchmark}" Kokkos::kokkos)
    endforeach()
endif()


#Serial
add_executable(serial_verification ${CMAKE_SOURCE_DIR}/src/serial_verification.cc)

