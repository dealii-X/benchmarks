

CXX=g++-14
CXXFLAGS=-Wall -O3 -march=native
CXXFLAGS+=-I./include/kernels -I./include -DCL_SILENCE_DEPRECATION
OPENCL_HPP_ROOT = $(HOME)/OpenCL-CLHPP

SYCL_CXX=icpx -fsycl
SYCL_CXXFLAGS=-O3 -xHOST

all: opencl_benchmark

opencl_benchmark: CXXFLAGS += -I$(OPENCL_HPP_ROOT)/include
opencl_benchmark: src/BK1/opencl_benchmark.cc
	$(CXX) $(CXXFLAGS) -framework OpenCL -o $@ $<

sycl_benchmark: src/BK1/sycl_benchmark.cc
	$(SYCL_CXX) $(SYCL_CXXFLAGS) -o $@ $<

openmp_benchmark: src/BK1/openmp_benchmark.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	$(RM) opencl_benchmark openmp_benchmark sycl_benchmark


.PHONY: run_openmp
run_openmp: opencl_benchmark
	./scripts/sweep_dofs_logspace.sh | awk '!/^#/'

.PHONY: run_opencl
run_opencl: opencl_benchmark
	./scripts/sweep_dofs_logspace.sh | awk '!/^#/' > apple_m2.txt
	python3 ./scripts/plot_bk1.py