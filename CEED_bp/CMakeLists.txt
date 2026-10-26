cmake_minimum_required(VERSION 3.13.4)

set(TARGET "bp3")
set(TARGET_SRC ${CMAKE_SOURCE_DIR}/src/${TARGET}.cc)

find_package(deal.II 9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET} CXX)

# Keep original bp3 target intact via deal.II autopilot
deal_ii_invoke_autopilot()
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Upgrade optimization flag from deal.II's default -O2 to -O3 for optimized targets
string(REPLACE "-O2" "-O3" DEAL_II_CXX_FLAGS_RELEASE "${DEAL_II_CXX_FLAGS_RELEASE}")

# Optimized benchmark executable (new files only; original bp3 files untouched)
add_executable(bp3_opt ${CMAKE_CURRENT_SOURCE_DIR}/src/bp3_opt.cc)
set_target_properties(bp3_opt PROPERTIES CXX_EXTENSIONS OFF)
# Place unversioned libdeal_II.so (*.so) before deal_ii_setup_target so
# nvcc_wrapper keeps it in $object_files ahead of static libkokkoscore.a,
# preventing duplicate Kokkos singleton instantiation & exit-time double-free.
find_library(DEAL_II_UNVERSIONED_SO NAMES deal_II
  PATHS "${DEAL_II_PATH}/${DEAL_II_LIBRARY_RELDIR}" NO_DEFAULT_PATH)
if(DEAL_II_UNVERSIONED_SO)
  target_link_libraries(bp3_opt "${DEAL_II_UNVERSIONED_SO}")
endif()
deal_ii_setup_target(bp3_opt)
target_include_directories(bp3_opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Automated numerical verification suite (deformed/curved meshes, convergence, MPI)
enable_testing()
add_executable(verify_bp3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_bp3.cc)
set_target_properties(verify_bp3 PROPERTIES CXX_EXTENSIONS OFF)
if(DEAL_II_UNVERSIONED_SO)
  target_link_libraries(verify_bp3 "${DEAL_II_UNVERSIONED_SO}")
endif()
deal_ii_setup_target(verify_bp3)
target_include_directories(verify_bp3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME verify_bp3 COMMAND verify_bp3)
