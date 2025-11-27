cmake_minimum_required(VERSION 3.0)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK5)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK5)

set(BK5_CUDA_BENCHMARKS cuda_benchmark templated_cuda_benchmark constexpr_cuda_benchmark)
set(BK5_KOKKOS_BENCHMARKS kokkos_benchmark templated_kokkos_benchmark)

if(Kokkos_ENABLE_CUDA)
    foreach(benchmark IN LISTS BK5_CUDA_BENCHMARKS)
        add_executable("${benchmark}_bk5" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
        TARGET_LINK_LIBRARIES("${benchmark}_bk5" Kokkos::kokkos)
        set_target_properties("${benchmark}_bk5" PROPERTIES OUTPUT_NAME ${benchmark})
    endforeach()
endif()


foreach(benchmark IN LISTS BK5_KOKKOS_BENCHMARKS)
    add_executable("${benchmark}_bk5" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}_bk5" Kokkos::kokkos)
    set_target_properties("${benchmark}_bk5" PROPERTIES OUTPUT_NAME ${benchmark})
endforeach()




#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK5/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK5/tests)

if(Kokkos_ENABLE_CUDA)
    add_executable(cuda_verification_bk5   ${CMAKE_SOURCE_DIR}/tests/BK5/cuda_verification.cc)
    set_target_properties(cuda_verification_bk5 PROPERTIES OUTPUT_NAME cuda_verification)
    TARGET_LINK_LIBRARIES(cuda_verification_bk5 Kokkos::kokkos)
endif()


add_executable(serial_verification_bk5 ${CMAKE_SOURCE_DIR}/tests/BK5/serial_verification.cc)
set_target_properties(serial_verification_bk5 PROPERTIES OUTPUT_NAME serial_verification)

add_executable(kokkos_verification_bk5 ${CMAKE_SOURCE_DIR}/tests/BK5/kokkos_verification.cc)
set_target_properties(kokkos_verification_bk5 PROPERTIES OUTPUT_NAME kokkos_verification)
TARGET_LINK_LIBRARIES(kokkos_verification_bk5 Kokkos::kokkos)


#Reports
if(Kokkos_ENABLE_CUDA)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK5/reports)

    add_custom_target(reportBK5
        COMMAND ncu -f --set full -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        COMMAND nsys profile --force-overwrite=true  -o reports/benchmark_3_3_3_400000_27000_1 cuda_benchmark 3 3 3 400000 27000 3 3 3 1
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/BK5
    )
endif()



