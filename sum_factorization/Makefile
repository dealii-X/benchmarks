

CXX=g++-14
CXXFLAGS=-O3 -march=native

CXXFLAGS+=-I./include/kernels -I./include -DCL_SILENCE_DEPRECATION

OPENCL_HPP_ROOT = $(HOME)/OpenCL-CLHPP

all: opencl_benchmark

opencl_benchmark: CXXFLAGS += -I$(OPENCL_HPP_ROOT)/include
opencl_benchmark: src/BK1/opencl_benchmark.cc
	$(CXX) $(CXXFLAGS) -framework OpenCL -o $@ $<


.PHONY: clean
clean:
	$(RM) opencl_benchmark


.PHONY: run_opencl
run_opencl: opencl_benchmark
	./scripts/sweep_dofs_logspace.sh | awk '!/^#/' > apple_m2.txt
	python3 ./scripts/plot_bk1.py