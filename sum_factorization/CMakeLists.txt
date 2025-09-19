cmake_minimum_required(VERSION 3.0)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

#BKs
add_subdirectory(src/BK1)
add_subdirectory(src/BK3)
add_subdirectory(src/BK5)

#Tests
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_executable(test_lagrange ${CMAKE_SOURCE_DIR}/tests/test_lagrange.cc)

#Post-processing directory
configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/plot/plot.py
    ${CMAKE_BINARY_DIR}/outputs/plot.py
    COPYONLY
)

#Roofline directory
file(COPY ${CMAKE_SOURCE_DIR}/scripts/roofline
     DESTINATION ${CMAKE_BINARY_DIR}/)