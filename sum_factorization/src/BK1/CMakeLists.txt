cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK1)

add_executable(kokkos_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/kokkos_benchmark.cc)
add_executable(cuda_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/cuda_benchmark.cc)
add_executable(templated_cuda_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/templated_cuda_benchmark.cc)
add_executable(templated_kokkos_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/templated_kokkos_benchmark.cc)
add_executable(output_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/output_benchmark.cc)

TARGET_LINK_LIBRARIES(kokkos_benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(cuda_benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(templated_cuda_benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(templated_kokkos_benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(output_benchmark Kokkos::kokkos)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/outputs)
configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/plotBK1.gp
    ${CMAKE_BINARY_DIR}/BK1/outputs/plot.gp
    COPYONLY
)


#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/tests)

add_executable(serial_verification ${CMAKE_SOURCE_DIR}/tests/BK1/serial_verification.cc)
add_executable(cuda_verification   ${CMAKE_SOURCE_DIR}/tests/BK1/cuda_verification.cc)
add_executable(kokkos_verification ${CMAKE_SOURCE_DIR}/tests/BK1/kokkos_verification.cc)

TARGET_LINK_LIBRARIES(cuda_verification Kokkos::kokkos)
TARGET_LINK_LIBRARIES(kokkos_verification Kokkos::kokkos)

#Reports
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK1/reports)

add_custom_target(reportBK1
    COMMAND ncu -f --set full -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
    COMMAND nsys profile --force-overwrite=true  -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/BK1
)