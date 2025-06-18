cmake_minimum_required(VERSION 3.0)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

#BK1
add_subdirectory(src/BK1)

#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_executable(test_lagrange ${CMAKE_SOURCE_DIR}/tests/test_lagrange.cc)


