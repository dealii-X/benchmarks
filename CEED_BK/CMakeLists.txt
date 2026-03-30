cmake_minimum_required(VERSION 3.20)

project(benchmark LANGUAGES CXX)
find_package(Kokkos REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

#BKs
add_subdirectory(src/BK1)
add_subdirectory(src/BK3)
add_subdirectory(src/BK5)
