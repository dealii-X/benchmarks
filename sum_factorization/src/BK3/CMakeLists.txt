cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)


if(Kokkos_ENABLE_CUDA)

    add_executable(cuda_benchmark_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/cuda_benchmark.cc)
    set_target_properties(cuda_benchmark_bk3 PROPERTIES OUTPUT_NAME cuda_benchmark)
    TARGET_LINK_LIBRARIES(cuda_benchmark_bk3 Kokkos::kokkos)

endif()


#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3/tests)

if(Kokkos_ENABLE_CUDA)
    add_executable(cuda_verification_bk3   ${CMAKE_SOURCE_DIR}/tests/BK3/cuda_verification.cc)
    set_target_properties(cuda_verification_bk3 PROPERTIES OUTPUT_NAME cuda_verification)
    TARGET_LINK_LIBRARIES(cuda_verification_bk3 Kokkos::kokkos)
endif()

add_executable(serial_verification_bk3 ${CMAKE_SOURCE_DIR}/tests/BK3/serial_verification.cc)
set_target_properties(serial_verification_bk3 PROPERTIES OUTPUT_NAME serial_verification)



