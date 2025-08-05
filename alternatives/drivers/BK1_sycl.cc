/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <CL/sycl.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cmath>
#include <array>
#include <tuple>

#include "timer.hpp"


// Use SYCL specialization constants to embed the polynomial order at runtime
const static sycl::specialization_id<std::array<unsigned int,3>> nm_id; // interpolation degree
const static sycl::specialization_id<std::array<unsigned int,3>> nq_id; // quadrature degree

//enum {
//    , BTHK_QP_1D,
//    , BwdTransHexKernel_QP_1D_3D_BLOCKS
//    , BwdTransHexKernel_QP_1D_3D_BLOCKS_SimpleMap
//}

template<typename T>
struct BwdTransHexKernel_QP_1D {

    const unsigned int nm0, nm1, nm2;
    const unsigned int nq0, nq1, nq2;
    const unsigned int nelmt;

    const T* d_basis0;
    const T* d_basis1;
    const T* d_basis2;

    const T* d_JxW;
    const T* d_in;
          T* d_out;

    // Shared memory equivalent
    //      T* shared;
    sycl::local_accessor<T> shared_mem;

    BwdTransHexKernel_QP_1D(
        const unsigned int nm0_, const unsigned int nm1_, const unsigned int nm2_,
        const unsigned int nq0_, const unsigned int nq1_, const unsigned int nq2_,
        const unsigned int nelmt_,
        const T *__restrict__ d_basis0_,
        const T *__restrict__ d_basis1_,
        const T *__restrict__ d_basis2_,
        const T *__restrict__ d_JxW_,
        const T *__restrict__ d_in_,
        T *__restrict__ d_out_,
        sycl::local_accessor<T> shared_mem_) :
        nm0(nm0_), nm1(nm1_), nm2(nm2_),
        nq0(nq0_), nq1(nq1_), nq2(nq2_), nelmt(nelmt_),
        d_basis0(d_basis0_), d_basis1(d_basis1_), d_basis2(d_basis2_),
        d_JxW(d_JxW_), d_in(d_in_), d_out(d_out_), shared_mem(shared_mem_) {}

//    shared{sycl::range<1>{nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + 
//                          nm0 * nm1 * nm2 + 
//                          nm1 * nm2 * nq0 + 
//                          nm2 * nq0 * nq1 + 
//                          nq0 * nq1 * nq2}, cgh} {}

    void operator()(sycl::nd_item<1> idx, sycl::kernel_handler kh) const {

        // These now shadow the struct ones
        const auto [nm0,nm1,nm2] = kh.get_specialization_constant<nm_id>();
        const auto [nq0,nq1,nq2] = kh.get_specialization_constant<nq_id>();

        T *s_basis0 = &shared_mem[0];
        T *s_basis1 = s_basis0 + nm0 * nq0;
        T *s_basis2 = s_basis1 + nm1 * nq1;
        T *s_wsp0 = s_basis2 + nm2 * nq2;
        T *s_wsp1 = s_wsp0 + nm0 * nm1 * nm2;
        T *s_wsp2 = s_wsp1 + nm1 * nm2 * nq0;
        T *s_wsp3 = s_wsp2 + nm2 * nq0 * nq1;


        //copy to shared memory
        for(unsigned int tid = idx.get_local_id(0); tid < nq0 * nm0; tid += idx.get_local_range(0))
        {
            s_basis0[tid] = d_basis0[tid];
        }

        for(unsigned int tid = idx.get_local_id(0); tid < nq1 * nm1; tid += idx.get_local_range(0))
        {
            s_basis1[tid] = d_basis1[tid];
        }

        for(unsigned int tid = idx.get_local_id(0); tid < nq2 * nm2; tid += idx.get_local_range(0))
        {
            s_basis2[tid] = d_basis2[tid];
        }

        unsigned int e = idx.get_group(0);

        while(e < nelmt)
        {
            //Copy inptr to s_wsp0
            for(unsigned int tid = idx.get_local_id(0); tid < nm0 * nm1 * nm2; tid += idx.get_local_range(0))
            {
                s_wsp0[tid] = d_in[nm0 * nm1 * nm2 * e + tid];
            }
            idx.barrier(sycl::access::fence_space::local_space);

            // direction 0  -> tid = p * nm1 * nm2 + j * nm2 + k (wsp1)
            for(unsigned int tid = idx.get_local_id(0); tid < nq0 * nm1 * nm2; tid += idx.get_local_range(0))
            {
                unsigned int p = tid / (nm1 * nm2);
                unsigned int j = (tid % (nm1 * nm2)) / nm2;
                unsigned int k = tid % nm2;

                T tmp = 0.0;
                for(unsigned int i = 0; i < nm0; ++i)
                {
                    tmp += s_wsp0[i * nm1 * nm2 + j * nm2 + k] * s_basis0[p * nm0 + i];
                }
                s_wsp1[p * nm1 * nm2 + j * nm2 + k] = tmp;
            }
            idx.barrier(sycl::access::fence_space::local_space);

            //direction 1 -> tid = q * nq0 * nm2 + p * nm2 + k  (wsp2)
            for(unsigned int tid = idx.get_local_id(0); tid < nq0 * nq1 * nm2; tid += idx.get_local_range(0))
            {
                unsigned int q = tid / (nq0 * nm2);
                unsigned int p = (tid % (nq0 * nm2)) / nm2;
                unsigned int k = tid % nm2;

                T tmp = 0.0;
                for(unsigned int j = 0; j < nm1; j++)
                {
                    tmp += s_wsp1[p * nm1 * nm2 + j * nm2 + k] * s_basis1[q * nm1 + j];
                }
                s_wsp2[q * nq0 * nm2 + p * nm2 + k] = tmp;
            }
            idx.barrier(sycl::access::fence_space::local_space);

            //direction 2 -> tid = p * nq1 * nq2 + q * nq2 + r   (wsp3)
            for(unsigned int tid = idx.get_local_id(0); tid < nq0 * nq1 * nq2; tid += idx.get_local_range(0))
            {
                unsigned int p = tid / (nq1 * nq2);
                unsigned int q = (tid % (nq1 * nq2)) / nq2;
                unsigned int r = tid % nq2;

                T tmp = 0.0;
                for(unsigned int k = 0; k < nm2; ++k)
                {
                    tmp += s_wsp2[q * nq0 * nm2 + p * nm2 + k] * s_basis2[r * nm2 + k];
                }
                s_wsp3[p * nq1 * nq2 + q * nq2 + r] = tmp;
            }
            idx.barrier(sycl::access::fence_space::local_space);


            //Copy s_wsp3 to outptr
            for(unsigned int tid = idx.get_local_id(0); tid < nq0 * nq1 * nq2; tid += idx.get_local_range(0))
            {
                d_out[e * nq0 * nq1 * nq2 + tid] = s_wsp3[tid];
            }
            idx.barrier(sycl::access::fence_space::local_space);

            e += idx.get_group_range(0);
        }
    }
};

// Global settings
int show_norm = -1; // -1 means uninitialized


template<typename T>
void run_test(
    sycl::queue &queue,
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
    const unsigned int numThreads, 
    const unsigned int threadsPerBlockX,
    const unsigned int threadsPerBlockY,
    const unsigned int threadsPerBlockZ,
    const unsigned int nelmt,
    const unsigned int ntests)
{

    unsigned int threadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
    const unsigned int numBlocks = numThreads / (std::min(nq0 * nq1 * nq2, threadsPerBlock));

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

    //Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, (T)0.0);

    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq0; p++) {
        for(unsigned int i = 0u; i < nm0; i++) {
            basis0[p * nm0 + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for(unsigned int q = 0u; q < nq1; q++) {
        for(unsigned int j = 0u; j < nm1; j++) {
            basis1[q * nm1 + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for(unsigned int r = 0u; r < nq2; r++) {
        for(unsigned int k = 0u; k < nm2; k++) {
            basis2[r * nm2 + k] = std::cos((T)(r * nm2 + k));
        }
    }

    const size_t size_JxW = nelmt * nq0 * nq1 * nq2;
    const size_t size_in = nelmt * nm0 * nm1 * nm2;
    const size_t size_out = nelmt * nq0 * nq1 * nq2;
    const size_t size_basis0 = nq0*nm0;
    const size_t size_basis1 = nq1*nm1;
    const size_t size_basis2 = nq2*nm2;

    const unsigned int ssize = 2 * nq0 * nq1 * nq2 + nm0 * nq0 + nm1 * nq1 + nm2 * nq2; //shared memory dynamic size


    T *d_JxW    = sycl::malloc_shared<T>(size_JxW, queue);
    T *d_in     = sycl::malloc_shared<T>(size_in, queue);
    T *d_out    = sycl::malloc_shared<T>(size_out, queue);
    T *d_basis0 = sycl::malloc_device<T>(size_basis0, queue);
    T *d_basis1 = sycl::malloc_device<T>(size_basis1, queue);
    T *d_basis2 = sycl::malloc_device<T>(size_basis2, queue);

    queue.copy(in.data(),d_in,size_in);
    queue.copy(JxW.data(),d_JxW,size_JxW);
    queue.copy(basis0.data(),d_basis0,size_basis0);
    queue.copy(basis1.data(),d_basis1,size_basis1);
    queue.copy(basis2.data(),d_basis2,size_basis2);
    queue.wait_and_throw();


    const size_t localSize = std::min(nq0 * nq1 * nq2, threadsPerBlock);
    const size_t globalSize = numBlocks * localSize;

    const sycl::nd_range<1> kernelRange{{globalSize},{localSize}};

    std::cout << "# global size = " << globalSize << '\n';
    std::cout << "# local size  = " << localSize << '\n';

    const std::array<unsigned int,3> nms{nm0,nm1,nm2};
    const std::array<unsigned int,3> nqs{nq0,nq1,nq2};

    double time_cl = std::numeric_limits<double>::max();
    Timer clTimer;

    for (unsigned int t = 0u; t < ntests; ++t)
    {
        clTimer.start();

        queue.submit([=](sycl::handler &cgh){

            // Work-group local memory
            sycl::local_accessor<T> shared_mem{{ssize},cgh};
            //auto shared = shared_mem.template get_multi_ptr<sycl::access::decorated::no>().get();
            //T* shared = &shared_mem[0];

            cgh.template set_specialization_constant<nm_id>(nms);
            cgh.template set_specialization_constant<nq_id>(nqs);

            BwdTransHexKernel_QP_1D<T> kernel(
                nm0,nm1,nm2,nq0,nq1,nq2,nelmt,
                d_basis0,d_basis1,d_basis2,d_JxW,d_in,d_out,shared_mem);

            cgh.parallel_for(kernelRange, kernel);

        }).wait_and_throw();

        clTimer.stop();
        time_cl = std::min(time_cl, clTimer.elapsedSeconds());
    }

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed;
    };

    // C++23
    // std::println("OpenCL -> nelmt = {} GDoF/s = {}", nelmt, dof_rate(time_cl));

    // mode, kernel_name, nelmt, GDoF/s
    std::printf("%s\t%s\t%u\t%f\n", 
        "static", "BwdTransHexKernel_QP_1D", nelmt, dof_rate(time_cl));

    if (show_norm) {
        std::vector<T> host_out(size_out);

        // 2. Copy data from device buffer d_out to host buffer out
        queue.copy(d_out,out.data(),out.size()).wait_and_throw();
        queue.copy(d_out,host_out.data(),host_out.size()).wait_and_throw();

        double normSqr{0.0};
        for (auto h : host_out) {
            normSqr += ((double) h) * ((double) h);
        }
        
        normSqr = 0;
        for(int i = 0; i < size_out; i++) {
            normSqr += d_out[i] * d_out[i];
        }
        //const T normSqr = norm2(host_out);

        //std::println("OpenCL kernel norm = {}", std::sqrt(normSqr));

        std::cout << "# SYCL kernel norm = " << std::sqrt(normSqr) << '\n';
    }

}


int main(int argc, char **argv){

    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    unsigned int numThreads         = (argc > 5) ? atoi(argv[5]) : nelmt * nq0 * nq1 * nq2 / 2;
    unsigned int threadsPerBlockX   = (argc > 6) ? atoi(argv[6]) : nq0;
    unsigned int threadsPerBlockY   = (argc > 7) ? atoi(argv[7]) : nq1;
    unsigned int threadsPerBlockZ   = (argc > 8) ? atoi(argv[8]) : nq2;
    unsigned int ntests             = (argc > 9) ? atoi(argv[9]) : 20u;

    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    if (show_norm == -1) {
        const char *env = getenv("SHOW_NORM");
        show_norm = (env && strcmp(env, "1") == 0) ? 1 : 0;
    }

    try {
        // This tries to create a queue on a GPU device if available,
        // otherwise it throws an exception.
        sycl::queue queue(sycl::default_selector_v);

        std::cout << "# Running on device: "
                  << queue.get_device().get_info<sycl::info::device::name>()
                  << "\n";

        // Now you can set kernel arguments and enqueue kernel as shown before
        run_test<float>(queue,
            nq0, nq1, nq2, nm0, nm1, nm2, numThreads, threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, nelmt, ntests);

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception  caught: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
