cmake_minimum_required(VERSION 3.0)

project(benchmark LANGUAGES CXX )

find_package(OpenMP REQUIRED)

option(BUILD_OPENCL "Build OpenCL benchmarks")
option(BUILD_SYCL   "Build SYCL benchmarks")
option(BUILD_TINYTC "Build TinyTC benchmarks")

set (CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/../sum_factorization/include)

add_executable(BK1_openmp drivers/BK1_openmp.cc)
target_link_libraries(BK1_openmp OpenMP::OpenMP_CXX)

add_executable(BK5_openmp drivers/BK5_openmp.cc)
target_link_libraries(BK5_openmp OpenMP::OpenMP_CXX)

if(BUILD_OPENCL)
	message(STATUS "OpenCL benchmarks are enabled")

	find_package(OpenCL REQUIRED)
	include_directories($ENV{OPENCL_CLHPP_ROOT}/include)

	if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
	  add_compile_definitions(CL_SILENCE_DEPRECATION)
	endif()

	add_executable(BK1_opencl drivers/BK1_opencl.cc)
	target_link_libraries(BK1_opencl PRIVATE OpenCL::OpenCL)

	add_executable(BK5_opencl drivers/BK5_opencl.cc)
	target_link_libraries(BK5_opencl PRIVATE OpenCL::OpenCL)

endif()

if(BUILD_TINYTC)
	message(STATUS "TinyTC benchmark is enabled")

	find_package(tinytc REQUIRED)
	find_package(tinytc_sycl REQUIRED)

	add_executable(tinytc_driver drivers/tinytc.cc)
        target_link_libraries(tinytc_driver PRIVATE tinytc::tinytc tinytc::tinytc_sycl)
        target_compile_options(tinytc_driver PRIVATE "-fsycl")
        target_link_options(tinytc_driver PRIVATE "-fsycl")
endif()
