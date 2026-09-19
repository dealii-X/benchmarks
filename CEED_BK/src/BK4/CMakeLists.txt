cmake_minimum_required(VERSION 3.20)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK4)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK4)

set(BK4_CUDA_BENCHMARKS mma_f64_m8n8k4_benchmark batched_mma_f64_m8n8k4_benchmark)
set(BK4_KOKKOS_BENCHMARKS templated_kokkos_benchmark batched_kokkos_benchmark)

if(Kokkos_ENABLE_CUDA)
    foreach(benchmark IN LISTS BK4_CUDA_BENCHMARKS)
        add_executable("${benchmark}_bk4" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
        TARGET_LINK_LIBRARIES("${benchmark}_bk4" Kokkos::kokkos)
        set_target_properties("${benchmark}_bk4" PROPERTIES OUTPUT_NAME ${benchmark})
    endforeach()
endif()

foreach(benchmark IN LISTS BK4_KOKKOS_BENCHMARKS)
    add_executable("${benchmark}_bk4" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}_bk4" Kokkos::kokkos)
    set_target_properties("${benchmark}_bk4" PROPERTIES OUTPUT_NAME ${benchmark})
endforeach()


#Serial
add_executable(serial_verification_bk4 ${CMAKE_CURRENT_SOURCE_DIR}/serial_verification.cc)
set_target_properties(serial_verification_bk4 PROPERTIES OUTPUT_NAME serial_verification)






