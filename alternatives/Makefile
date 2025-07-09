

CXX=g++-14
CXXFLAGS=-Wall -O3 -march=native
CXXFLAGS+=-I./src/ -I../sum_factorization/include

OPENCL_HPP_ROOT = $(HOME)/OpenCL-CLHPP
OPENCL_INC = -I$(OPENCL_HPP_ROOT)/include
OPENCL_LDFLAGS = -framework OpenCL

SYCL_CXX=icpx -fsycl
SYCL_CXXFLAGS=-O3 -xHOST

uname_s := $(shell uname -s)

ifeq ($(uname_s),Darwin)
CXXFLAGS += -DCL_SILENCE_DEPRECATION
endif

all: opencl_benchmark

opencl_benchmark: drivers/opencl_benchmark.cc
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(OPENCL_LDFLAGS) -o $@ $<

sycl_benchmark: drivers/sycl_benchmark.cc
	$(SYCL_CXX) $(SYCL_CXXFLAGS) -o $@ $<

openmp_benchmark: drivers/openmp_benchmark.cpp
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