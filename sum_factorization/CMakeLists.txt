cmake_minimum_required(VERSION 3.0)

project(benchmark LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/kernels)

add_executable(benchmark src/benchmark.cc)
add_executable(block_dimension_benchmark src/block_dimension_benchmark.cc)


find_package(Kokkos REQUIRED)
TARGET_LINK_LIBRARIES(benchmark Kokkos::kokkos)
TARGET_LINK_LIBRARIES(block_dimension_benchmark Kokkos::kokkos)



option(ENABLE_TESTS "Enable building of test programs" ON)
if(ENABLE_TESTS)
    message(STATUS "building tests ...")
    add_subdirectory(tests)
endif()


set(REPORT_DIR ${CMAKE_BINARY_DIR}/reports)
file(MAKE_DIRECTORY ${REPORT_DIR})

add_custom_target(report
COMMAND ncu -f --set full -o ${REPORT_DIR}/benchmark_4_4_4_20480_256_2000_1 benchmark 4 4 4 20480 256 2000 1
COMMAND nsys profile --force-overwrite=true  -o ${REPORT_DIR}/benchmark_4_4_4_20480_256_2000_1 benchmark 4 4 4 20480 256 2000 1
WORKING_DIRECTORY ${CMAKE_BUILD_DIR}
)
