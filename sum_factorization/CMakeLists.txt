cmake_minimum_required(VERSION 3.0)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(benchmark src/benchmark.cc)
add_executable(block_dimension_benchmark src/block_dimension_benchmark.cc)
add_executable(benchmark_output src/benchmark_output.cc)

TARGET_LINK_LIBRARIES(benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(block_dimension_benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(benchmark_output Kokkos::kokkos)


option(ENABLE_TESTS "Enable building of test programs" ON)
if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/reports)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/outputs)
configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/plot.gp
    ${CMAKE_BINARY_DIR}/outputs/plot.gp
    COPYONLY
)

add_custom_target(report
    COMMAND ncu -f --set full -o reports/benchmark_4_4_4_20480_256_2000_1 benchmark 4 4 4 20480 256 2000 1
    COMMAND nsys profile --force-overwrite=true  -o reports/benchmark_4_4_4_20480_256_2000_1 benchmark 4 4 4 20480 256 2000 1
    WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
)
