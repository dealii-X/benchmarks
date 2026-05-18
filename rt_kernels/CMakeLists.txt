cmake_minimum_required(VERSION 3.20)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/kokkos_kernels)
include_directories(${CMAKE_SOURCE_DIR}/include/serial_kernels)


set(KOKKOS_BENCHMARKS kokkosMassBenchmark)

foreach(benchmark IN LISTS KOKKOS_BENCHMARKS)
    add_executable("${benchmark}" "${CMAKE_CURRENT_SOURCE_DIR}/src/kokkos/${benchmark}.cc")
    TARGET_LINK_LIBRARIES("${benchmark}" Kokkos::kokkos)
endforeach()


#Serial
add_executable(serial_verification ${CMAKE_SOURCE_DIR}/src/serial_verification.cc)

