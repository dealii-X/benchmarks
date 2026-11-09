cmake_minimum_required(VERSION 3.20)

#Benchmark executables
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/BK2)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/BK2)

set(BK2_KOKKOS_BENCHMARKS templated_kokkos_benchmark templated_batched_kokkos_benchmark)

foreach(benchmark IN LISTS BK2_KOKKOS_BENCHMARKS)
    add_executable("${benchmark}_bk2" "${CMAKE_CURRENT_SOURCE_DIR}/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}_bk2" Kokkos::kokkos)
    set_target_properties("${benchmark}_bk2" PROPERTIES OUTPUT_NAME ${benchmark})
endforeach()

    

#Serial
add_executable(serial_verification_bk2 ${CMAKE_CURRENT_SOURCE_DIR}/serial_verification.cc)
set_target_properties(serial_verification_bk2 PROPERTIES OUTPUT_NAME serial_verification)