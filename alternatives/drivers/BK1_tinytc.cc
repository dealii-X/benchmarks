/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages 
*/

#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>

#include <iostream>
#include <fstream>
#include <cstdio>
#include <numeric>
#include <vector>
#include <cmath>
#include <array>
#include <tuple>


class Timer {
public:
    void start(){
        m_StartTime = m_clock::now();
        m_bRunning  = true;
    }

    void stop(){
        m_EndTime  = m_clock::now();
        m_bRunning = false;
    }

    double elapsedNanoseconds(){
        auto endTime = m_bRunning ? m_clock::now() : m_EndTime;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - m_StartTime).count();
    }

    double elapsedSeconds(){
        return elapsedNanoseconds() / 1.0e9;
    }

private:
    using m_clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<m_clock> m_StartTime{};
    std::chrono::time_point<m_clock> m_EndTime{};
    bool m_bRunning = false;
};


template<typename T, int nq0, int nq1, int nq2>
void run_test(
    sycl::queue &queue,
    sycl::kernel_bundle<sycl::bundle_state::executable>& bundle,
    const unsigned int nelmt,
    const unsigned int ntests,
    bool show_norm)
{
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    std::vector<T> basis0(nm0 * nq0), basis1(nm1 * nq1), basis2(nm2 * nq2);

    //Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nm0 * nm1 * nm2, (T)0.0);

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

    const size_t size_JxW = JxW.size();
    const size_t size_in = in.size();
    const size_t size_out = out.size();
    const size_t size_basis0 = basis0.size();
    const size_t size_basis1 = basis1.size();
    const size_t size_basis2 = basis2.size();

    T *d_JxW    = sycl::malloc_device<T>(size_JxW, queue);
    T *d_in     = sycl::malloc_device<T>(size_in, queue);
    T *d_out    = sycl::malloc_device<T>(size_out, queue);
    T *d_basis0 = sycl::malloc_device<T>(size_basis0, queue);
    T *d_basis1 = sycl::malloc_device<T>(size_basis1, queue);
    T *d_basis2 = sycl::malloc_device<T>(size_basis2, queue);

    queue.copy(in.data(),d_in,size_in);
    queue.copy(JxW.data(),d_JxW,size_JxW);
    queue.copy(basis0.data(),d_basis0,size_basis0);
    queue.copy(basis1.data(),d_basis1,size_basis1);
    queue.copy(basis2.data(),d_basis2,size_basis2);
    queue.wait_and_throw();

//    const size_t localSize = std::min(nq0 * nq1 * nq2, (int) threadsPerBlock);
//    const size_t globalSize = numBlocks * localSize;

//    const sycl::nd_range<1> kernelRange{{globalSize},{localSize}};

//    const std::array<unsigned int,3> nms{nm0,nm1,nm2};
//    const std::array<unsigned int,3> nqs{nq0,nq1,nq2};

//    const std::array<int,3> nq_sf{nq0,nq1,nq2};

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed;
    };

    {
        const size_t howmany = nelmt;
        auto bk1_kernel = tinytc::create_kernel(bundle, "sum_factorization");
        auto exe_range = tinytc::get_execution_range(
            bk1_kernel, sycl::range<3u>{1u,1u,howmany});

        double time_cl = std::numeric_limits<double>::max();
        Timer clTimer;

        for (unsigned int t = 0u; t < ntests; ++t)
        {
            clTimer.start();

            queue.submit([=](sycl::handler &h){
                h.set_args(d_basis0,d_basis1,d_basis2,d_JxW,howmany,d_in,howmany,d_out,howmany);
                h.parallel_for(exe_range,bk1_kernel);
            }).wait_and_throw();

            clTimer.stop();
            time_cl = std::min(time_cl, clTimer.elapsedSeconds());
        }

        std::cout << "# TinyTC -> nelmt " << nelmt << " GDoF/s = " << dof_rate(time_cl) << '\n';

    }

    if (show_norm) {

        // 2. Copy data from device buffer d_out to host buffer out
        queue.copy(d_out,out.data(),out.size()).wait_and_throw();

        double normSqr{0.0};
        for (auto h : out) {
            normSqr += ((double) h) * ((double) h);
        }

        std::cout << "# TinyTC kernel norm = " << std::sqrt(normSqr) << '\n';
    }

}


int main(int argc, char **argv){

    const unsigned int nq0 = 4;
    const unsigned int nq1 = 4;
    const unsigned int nq2 = 4;

    unsigned int nelmt              = (argc > 1) ? atoi(argv[1]) : 2 << 18;
    unsigned int ntests             = (argc > 2) ? atoi(argv[2]) : 5u;

    const char *env = std::getenv("SHOW_NORM");
    bool show_norm = (env && strcmp(env, "1") == 0);

    try {
        // This tries to create a queue on a GPU device if available,
        // otherwise it throws an exception.
        sycl::queue queue(sycl::default_selector_v);

        auto device = queue.get_device();

        std::cout << "# Device name: " << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "#   Max Compute Units (EUs): " << device.get_info<sycl::info::device::max_compute_units>() << '\n';
        std::cout << "#   Max Work Group Size: " << device.get_info<sycl::info::device::max_work_group_size>() << '\n';
        std::cout << "#   Sub-group Sizes: ";
        for (const auto &s : device.get_info<sycl::info::device::sub_group_sizes>()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;

        auto info = tinytc::create_core_info(device);
        tinytc::set_core_features(info.get(), tinytc_core_feature_flag_large_register_file);

        auto ctx = tinytc::create_compiler_context();
        tinytc::set_error_reporter(ctx.get(), [](char const *what, const tinytc_location_t *, void *) {
           std::cerr << what << std::endl;
        });


        // Parse program bundle
        auto prog = tinytc::parse_file("sum_factorization.ir", ctx.get());

        auto bin = tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());

        auto bundle = tinytc::create_kernel_bundle(queue.get_context(), device, bin.get());

        // Now you can set kernel arguments and enqueue kernel as shown before
        run_test<float,nq0,nq1,nq2>(queue, bundle, nelmt, ntests, show_norm);

    } catch (tinytc::status const& st) {
        std::cerr << "tinytc: Error (" << static_cast<int>(st) << ")" << std::endl;
        return 1;
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
