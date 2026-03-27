cmake_minimum_required(VERSION 3.13.4)

set(TARGET "bp3")

set(TARGET_SRC ${CMAKE_SOURCE_DIR}/src/${TARGET}.cc)


include_directories(${TARGET} PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET} CXX)


# 6. Autopilot handles linking, flags (-O3), and Kokkos/MPI setup
deal_ii_invoke_autopilot()

