cmake_minimum_required(VERSION 3.0)

add_executable(test_lagrange test_lagrange.cc)
add_executable(verification verification.cc)

TARGET_LINK_LIBRARIES(verification Kokkos::kokkos)