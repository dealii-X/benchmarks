cmake_minimum_required(VERSION 3.20)

project(Kokkos_MPI)

find_package(MPI REQUIRED)
find_package(Kokkos REQUIRED)

add_executable(p-halox phalox.cc)

target_link_libraries(p-halox PRIVATE Kokkos::kokkos)
target_link_libraries(p-halox PRIVATE MPI::MPI_CXX)
