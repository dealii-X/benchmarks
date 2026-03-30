cmake_minimum_required(VERSION 3.20)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK3)

set(BK3_CUDA_BENCHMARKS templated_cuda_benchmark)
set(BK3_KOKKOS_BENCHMARKS templated_kokkos_benchmark templated_batched_kokkos_benchmark)

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


#Serial
add_executable(serial_verification_bk3 ${CMAKE_CURRENT_SOURCE_DIR}/serial_verification.cc)
set_target_properties(serial_verification_bk3 PROPERTIES OUTPUT_NAME serial_verification)






