cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)

set(BK3_CUDA_BENCHMARKS cuda_benchmark templated_cuda_benchmark)
set(BK3_KOKKOS_BENCHMARKS kokkos_benchmark templated_kokkos_benchmark)

if(Kokkos_ENABLE_CUDA)
    foreach(benchmark IN LISTS BK3_CUDA_BENCHMARKS)
        add_executable("${benchmark}_bk3" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
        TARGET_LINK_LIBRARIES("${benchmark}_bk3" Kokkos::kokkos)
        set_target_properties("${benchmark}_bk3" PROPERTIES OUTPUT_NAME ${benchmark})
    endforeach()
endif()


foreach(benchmark IN LISTS BK3_KOKKOS_BENCHMARKS)
    add_executable("${benchmark}_bk3" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}_bk3" Kokkos::kokkos)
    set_target_properties("${benchmark}_bk3" PROPERTIES OUTPUT_NAME ${benchmark})
endforeach()




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

#Reports
if(Kokkos_ENABLE_CUDA)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3/reports)

    add_custom_target(reportBK3
        COMMAND ncu -f --set full -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        COMMAND nsys profile --force-overwrite=true  -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)
endif()



