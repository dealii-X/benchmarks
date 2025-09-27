// Copyright (C) 2025 Leibniz-Rechenzentrum
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/*
Input Parameters for Kernels;
nm : Number of the node on each direction
nq : Number of the gauss points on each direction
nelmt : Number of finite elements
basis : 1D basis functions on each dimension (l0, l1 etc.)

wsp : intermediate storages
*/

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>
#include <tuple>
#include <vector>

class Timer {
   public:
    void start() {
        m_StartTime = m_clock::now();
        m_bRunning = true;
    }

    void stop() {
        m_EndTime = m_clock::now();
        m_bRunning = false;
    }

    double elapsedNanoseconds() {
        auto endTime = m_bRunning ? m_clock::now() : m_EndTime;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                                    m_StartTime)
            .count();
    }

    double elapsedSeconds() { return elapsedNanoseconds() / 1.0e9; }

   private:
    using m_clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<m_clock> m_StartTime{};
    std::chrono::time_point<m_clock> m_EndTime{};
    bool m_bRunning = false;
};

template <typename T>
void run_test(sycl::queue &queue, const int nq0, const int nq1, const int nq2,
              const std::int64_t nelmt, const std::int64_t ntests,
              bool show_norm, bool init_random = false) {
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    const auto nq0_pad = [&]() {
        int nq0_pad = 1;
        while (nq0_pad < nq0) {
            nq0_pad *= 2;
        }
        return nq0_pad;
    }();
    const auto nq1_pad = nq0_pad;
    const auto nq2_pad = nq0_pad;

    std::vector<T> basis0(nq0_pad * nq0_pad, T{0}),
        basis1(nq1_pad * nq1_pad, T{0}), basis2(nq2_pad * nq2_pad, T{0});

    // Initialize the input and output arrays
    std::vector<T> JxW(nelmt * nq0 * nq1 * nq2, (T)1.0);
    std::vector<T> in(nelmt * nm0 * nm1 * nm2, (T)3.0);
    std::vector<T> out(nelmt * nm0 * nm1 * nm2, (T)0.0);

    if (init_random) {
        auto rd = std::random_device{};
        auto gen = std::minstd_rand(rd());
        auto dis = std::uniform_real_distribution<>(-1.0, 1.0);
        for (auto &j : JxW) {
            j = dis(gen);
        }
        for (auto &i : in) {
            i = dis(gen);
        }
    }

    // Initialization of basis functions
    for (unsigned int p = 0u; p < nq0; p++) {
        for (unsigned int i = 0u; i < nm0; i++) {
            basis0[p * nq0_pad + i] = std::cos((T)(p * nm0 + i));
        }
    }
    for (unsigned int q = 0u; q < nq1; q++) {
        for (unsigned int j = 0u; j < nm1; j++) {
            basis1[q * nq1_pad + j] = std::cos((T)(q * nm1 + j));
        }
    }
    for (unsigned int r = 0u; r < nq2; r++) {
        for (unsigned int k = 0u; k < nm2; k++) {
            basis2[r * nq2_pad + k] = std::cos((T)(r * nm2 + k));
        }
    }

    const size_t size_JxW = JxW.size();
    const size_t size_in = in.size();
    const size_t size_out = out.size();
    const size_t size_basis0 = basis0.size();
    const size_t size_basis1 = basis1.size();
    const size_t size_basis2 = basis2.size();

    T *d_JxW = sycl::malloc_device<T>(size_JxW, queue);
    T *d_in = sycl::malloc_device<T>(size_in, queue);
    T *d_out = sycl::malloc_device<T>(size_out, queue);
    T *d_basis0 = sycl::malloc_device<T>(size_basis0, queue);
    T *d_basis1 = sycl::malloc_device<T>(size_basis1, queue);
    T *d_basis2 = sycl::malloc_device<T>(size_basis2, queue);

    queue.copy(in.data(), d_in, size_in);
    queue.copy(JxW.data(), d_JxW, size_JxW);
    queue.copy(basis0.data(), d_basis0, size_basis0);
    queue.copy(basis1.data(), d_basis1, size_basis1);
    queue.copy(basis2.data(), d_basis2, size_basis2);
    queue.wait_and_throw();

    // Performance in GDoF/s
    auto dof_rate = [=](double elapsed) {
        return 1.0e-9 * nelmt * nm0 * nm1 * nm2 / elapsed;
    };
    auto byte_rate = [=](double elapsed) {
        return 1.0e-9 * sizeof(float) * nelmt *
               (2 * nm0 * nm1 * nm2 + nq0 * nq1 * nq2) / elapsed;
    };

    {
        auto info = tinytc::create_core_info(queue.get_device());
        // tinytc::set_core_features(info.get(),
        // tinytc_core_feature_flag_large_register_file);

        auto ctx = tinytc::create_compiler_context();
        tinytc::set_error_reporter(
            ctx.get(), [](char const *what, const tinytc_location_t *, void *) {
                std::cerr << what << std::endl;
            });

        // Parse program bundle

        char const *kernel_name = "sum_factorization_block_pre";
        constexpr std::size_t block_size = 16;
        // char const *kernel_name = "sum_factorization";
        // constexpr std::size_t block_size = 1;
        auto code_stream = std::ifstream("sum_factorization.ir");
        auto code_template =
            std::string(std::istreambuf_iterator<char>(code_stream),
                        std::istreambuf_iterator<char>());

        const auto wgs = [&]() {
            switch (nq0) {
                case 2:
                case 3:
                    return 32;
                case 4:
                    return 64;
                case 5:
                    return 128;
                case 6:
                    return 256;
                case 7:
                case 8:
                    return 512;
                default:
                    break;
            }
            return 32;
        }();
        auto code = (std::ostringstream{} << "$P = " << nm0 << "\n"
                                          << "$Q_pad = " << nq0_pad << "\n"
                                          << "$B = " << block_size << "\n"
                                          << "$wgs = " << wgs << "\n"
                                          << code_template)
                        .str();
        auto prog = tinytc::parse_string(code, ctx.get());

        auto bin =
            tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());

        auto bundle = tinytc::create_kernel_bundle(
            queue.get_context(), queue.get_device(), bin.get());

        auto bk1_kernel = tinytc::create_kernel(bundle, kernel_name);
        auto exe_range = tinytc::get_execution_range(
            bk1_kernel,
            sycl::range<3u>{1u, 1u,
                            static_cast<std::size_t>(nelmt) / block_size});

        double time_cl = std::numeric_limits<double>::max();
        Timer clTimer;

        for (unsigned int t = 0u; t < ntests; ++t) {
            clTimer.start();

            queue
                .submit([=](sycl::handler &h) {
                    h.set_args(d_basis0, d_basis1, d_basis2, d_JxW, nelmt, d_in,
                               nelmt, d_out, nelmt);
                    h.parallel_for(exe_range, bk1_kernel);
                })
                .wait_and_throw();

            clTimer.stop();
            time_cl = std::min(time_cl, clTimer.elapsedSeconds());
        }

        std::cout << "# TinyTC -> nelmt " << nelmt
                  << " GDoF/s = " << dof_rate(time_cl) << '\n';
        std::cout << "# TinyTC -> nelmt " << nelmt
                  << " GB/s = " << byte_rate(time_cl) << '\n';
    }

    if (show_norm) {
        // 2. Copy data from device buffer d_out to host buffer out
        queue.copy(d_out, out.data(), out.size()).wait_and_throw();

        double normSqr{0.0};
        for (auto h : out) {
            normSqr += ((double)h) * ((double)h);
        }

        std::cout << "# SYCL kernel norm = " << std::sqrt(normSqr) << '\n';
    }
}

int main(int argc, char **argv) {
    const unsigned int nq0 = argc > 1 ? atoi(argv[1]) : 4;
    const unsigned int nq1 = nq0;
    const unsigned int nq2 = nq0;

    unsigned int nelmt = argc > 2 ? atoi(argv[2]) : 2 << 18;
    unsigned int ntests = argc > 3 ? atoi(argv[3]) : 5u;

    const char *env = getenv("SHOW_NORM");
    const bool show_norm = (env && strcmp(env, "1") == 0);

    const char *env_rnd = getenv("INIT_RANDOM");
    const bool init_random = (env_rnd && strcmp(env_rnd, "1") == 0);

    try {
        // This tries to create a queue on a GPU device if available,
        // otherwise it throws an exception.
        sycl::queue queue(sycl::default_selector_v);

        auto device = queue.get_device();

        std::cout << "# Device name: "
                  << device.get_info<sycl::info::device::name>() << '\n';
        std::cout << "#   Max Compute Units (EUs): "
                  << device.get_info<sycl::info::device::max_compute_units>()
                  << '\n';
        std::cout << "#   Max Work Group Size: "
                  << device.get_info<sycl::info::device::max_work_group_size>()
                  << '\n';
        std::cout << "#   Sub-group Sizes: ";
        for (const auto &s :
             device.get_info<sycl::info::device::sub_group_sizes>()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;

        // Now you can set kernel arguments and enqueue kernel as shown
        // before
        run_test<float>(queue, nq0, nq1, nq2, nelmt, ntests, show_norm,
                        init_random);

    } catch (tinytc::status const &st) {
        std::cerr << "tinytc: Error (" << static_cast<int>(st) << ")"
                  << std::endl;
        return 1;
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
