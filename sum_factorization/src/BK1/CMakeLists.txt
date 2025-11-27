cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)

set(BK1_CUDA_BENCHMARKS cuda_benchmark templated_cuda_benchmark cuda_mma_benchmark templated_cuda_mma_benchmark)
set(BK1_KOKKOS_BENCHMARKS kokkos_benchmark templated_kokkos_benchmark)

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
    

#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/tests)

if(Kokkos_ENABLE_CUDA)
    add_executable(cuda_verification   ${CMAKE_SOURCE_DIR}/tests/BK1/cuda_verification.cc)
    TARGET_LINK_LIBRARIES(cuda_verification Kokkos::kokkos)
endif()

add_executable(serial_verification ${CMAKE_SOURCE_DIR}/tests/BK1/serial_verification.cc)
add_executable(kokkos_verification ${CMAKE_SOURCE_DIR}/tests/BK1/kokkos_verification.cc)

TARGET_LINK_LIBRARIES(kokkos_verification Kokkos::kokkos)


#Reports
if(Kokkos_ENABLE_CUDA)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/reports)

    add_custom_target(reportBK1
        COMMAND ncu -f --set full -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        COMMAND nsys profile --force-overwrite=true  -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)

endif()
