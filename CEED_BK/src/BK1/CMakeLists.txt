cmake_minimum_required(VERSION 3.20)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)

set(BK1_CUDA_BENCHMARKS templated_cuda_benchmark)
set(BK1_KOKKOS_BENCHMARKS templated_kokkos_benchmark templated_batched_kokkos_benchmark)

if(Kokkos_ENABLE_CUDA)
    foreach(benchmark IN LISTS BK1_CUDA_BENCHMARKS)
        add_executable(${benchmark} "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
        TARGET_LINK_LIBRARIES(${benchmark} Kokkos::kokkos)
    endforeach()
endif()

foreach(benchmark IN LISTS BK1_KOKKOS_BENCHMARKS)
    add_executable(${benchmark} "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
    TARGET_LINK_LIBRARIES(${benchmark} Kokkos::kokkos)
endforeach()
    

#Serial
add_executable(serial_verification ${CMAKE_CURRENT_SOURCE_DIR}/serial_verification.cc)