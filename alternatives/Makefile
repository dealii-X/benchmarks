

CXX=g++-15
CXXFLAGS=-std=c++17 -Wall -O3 -mcpu=native -fopenmp -funroll-loops
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

all: opencl_benchmark bk5_opencl

opencl_benchmark: drivers/opencl_benchmark.cc
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(OPENCL_LDFLAGS) -o $@ $<

bk5_opencl: drivers/bk5_opencl.cc
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) $(OPENCL_LDFLAGS) -o $@ $<

sycl_benchmark: drivers/sycl_benchmark.cc
	$(SYCL_CXX) $(SYCL_CXXFLAGS) -o $@ $<

openmp_benchmark: drivers/openmp_benchmark.cc bk1.o
	$(CXX) $(CXXFLAGS) -o $@ $^

FC=gfortran
#FFLAGS=-Wall -O2 -ftree-vectorize -funroll-loops -ffast-math -fopenmp -mcpu=native
FFLAGS=-Wall -O3 -funroll-loops -fopenmp -mcpu=native

bk1.o: src/bk1.F90
	$(FC) -cpp $(FFLAGS) -c $<

.PHONY: clean
clean:
	$(RM) *.o opencl_benchmark openmp_benchmark sycl_benchmark


.PHONY: run_openmp
run_openmp: opencl_benchmark
	./scripts/sweep_dofs_logspace.sh | awk '!/^#/'

.PHONY: run_opencl
run_opencl: opencl_benchmark
	./scripts/sweep_dofs_logspace.sh | awk '!/^#/' > apple_m2.txt
	python3 ./scripts/plot_bk1.py
