cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)


if(Kokkos_ENABLE_CUDA)

    add_executable(cuda_benchmark_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_benchmark.cc)
    set_target_properties(cuda_benchmark_bk3 PROPERTIES OUTPUT_NAME cuda_benchmark)
    TARGET_LINK_LIBRARIES(cuda_benchmark_bk3 Kokkos::kokkos)

    add_executable(templated_cuda_benchmark_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/templated_cuda_benchmark.cc)
    set_target_properties(templated_cuda_benchmark_bk3 PROPERTIES OUTPUT_NAME templated_cuda_benchmark)
    TARGET_LINK_LIBRARIES(templated_cuda_benchmark_bk3 Kokkos::kokkos)

endif()

add_executable(kokkos_benchmark_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_benchmark.cc)
set_target_properties(kokkos_benchmark_bk3 PROPERTIES OUTPUT_NAME kokkos_benchmark)
TARGET_LINK_LIBRARIES(kokkos_benchmark_bk3 Kokkos::kokkos)

add_executable(templated_kokkos_benchmark_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/templated_kokkos_benchmark.cc)
set_target_properties(templated_kokkos_benchmark_bk3 PROPERTIES OUTPUT_NAME templated_kokkos_benchmark)
TARGET_LINK_LIBRARIES(templated_kokkos_benchmark_bk3 Kokkos::kokkos)


#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3/tests)

if(Kokkos_ENABLE_CUDA)
    add_executable(cuda_verification_bk3   ${CMAKE_SOURCE_DIR}/tests/BK3/cuda_verification.cc)
    set_target_properties(cuda_verification_bk3 PROPERTIES OUTPUT_NAME cuda_verification)
    TARGET_LINK_LIBRARIES(cuda_verification_bk3 Kokkos::kokkos)
endif()

add_executable(kokkos_verification_bk3 ${CMAKE_SOURCE_DIR}/tests/BK3/kokkos_verification.cc)
set_target_properties(kokkos_verification_bk3 PROPERTIES OUTPUT_NAME kokkos_verification)
TARGET_LINK_LIBRARIES(kokkos_verification_bk3 Kokkos::kokkos)

add_executable(serial_verification_bk3 ${CMAKE_SOURCE_DIR}/tests/BK3/serial_verification.cc)
set_target_properties(serial_verification_bk3 PROPERTIES OUTPUT_NAME serial_verification)



