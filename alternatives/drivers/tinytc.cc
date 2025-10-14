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
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <tinytc/tinytc.hpp>
#include <tinytc/tinytc_sycl.hpp>
#include <tuple>
#include <vector>

#include "reference_kernels.hh"

using namespace std::chrono;

class sycl_deleter {
   public:
    sycl_deleter(sycl::queue q) : q_(std::move(q)) {}
    void operator()(void *ptr) { sycl::free(ptr, q_); }

   private:
    sycl::queue q_;
};

template <typename T>
auto make_constant_vec(std::size_t size, T value) -> std::vector<T> {
    return std::vector<T>(size, value);
}
template <typename T>
auto make_random_vec(std::size_t size) -> std::vector<T> {
    auto vec = std::vector<T>(size);
    auto rd = std::random_device{};
    auto gen = std::minstd_rand(rd());
    auto dis = std::uniform_real_distribution<>(-1.0, 1.0);
    for (auto &v : vec) {
        v = dis(gen);
    }
    return vec;
}
template <typename T>
auto make_constant_or_random_vec(std::size_t size, T value, bool init_random)
    -> std::vector<T> {
    return init_random ? make_random_vec<T>(size)
                       : make_constant_vec<T>(size, value);
}
template <typename T>
auto make_basis_vec(int rows, int cols, int stride) {
    auto basis = std::vector<T>(stride * stride, T{0});
    for (unsigned int j = 0u; j < cols; j++) {
        for (unsigned int i = 0u; i < rows; i++) {
            basis[j * stride + i] = std::cos((T)(j * rows + i));
        }
    }
    return basis;
}

template <typename T>
auto make_d_buf(std::vector<T> const &h_buf, sycl::queue q)
    -> std::unique_ptr<T, sycl_deleter> {
    auto d_buf = std::unique_ptr<T, sycl_deleter>(
        sycl::malloc_device<T>(h_buf.size(), q), sycl_deleter{q});
    q.copy(h_buf.data(), d_buf.get(), h_buf.size());
    return d_buf;
}

auto bk1_5_work_group_size(const int nq0) {
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
}

auto pad_nq(const int nq) {
    int nq_pad = 1;
    while (nq_pad < nq) {
        nq_pad *= 2;
    }
    return nq_pad;
}

struct kernel_params {
    int nq0;
    int nq0_pad;
    int block_size;
    int wgs;
};
auto make_kernel(char const *path, char const *kernel_name,
                 kernel_params const &params, sycl::queue queue) {
    auto info = tinytc::create_core_info(queue.get_device());
    auto ctx = tinytc::create_compiler_context();
    tinytc::set_error_reporter(
        ctx.get(), [](char const *what, const tinytc_location_t *, void *) {
            std::cerr << what << std::endl;
        });

    auto code_stream = std::ifstream(path);
    auto code_template =
        std::string(std::istreambuf_iterator<char>(code_stream),
                    std::istreambuf_iterator<char>());

    auto code = (std::ostringstream{} << "$Q = " << params.nq0 << "\n"
                                      << "$Q_pad = " << params.nq0_pad << "\n"
                                      << "$B = " << params.block_size << "\n"
                                      << "$wgs = " << params.wgs << "\n"
                                      << code_template)
                    .str();
    auto prog = tinytc::parse_string(code, ctx.get());
    auto bin = tinytc::compile_to_spirv_and_assemble(prog.get(), info.get());

    auto bundle = tinytc::create_kernel_bundle(queue.get_context(),
                                               queue.get_device(), bin.get());

    return tinytc::create_kernel(bundle, kernel_name);
}

template <typename T>
void show_error_norm(std::vector<T> const &out, std::vector<T> const &out_ref) {
    double diff2{0.0}, norm2{0.0};
    for (std::size_t i = 0; i < out.size(); ++i) {
        auto diff = out_ref[i] - out[i];
        diff2 += diff * diff;
        norm2 += out_ref[i] * out_ref[i];
    }

    auto diff = std::sqrt(diff2);
    auto norm = std::sqrt(norm2);
    constexpr auto eps = std::numeric_limits<T>::epsilon();
    std::cout << "# Abs error = " << diff << ", " << diff / eps << "ϵ"
              << std::endl;
    std::cout << "# Rel error = " << diff / norm << ", " << diff / (norm * eps)
              << "ϵ" << std::endl;
}

template <typename T>
void run_bk1(sycl::queue &queue, const int nq0, const int nq1, const int nq2,
             const std::int64_t nelmt, const std::int64_t ntests,
             bool show_norm, bool init_random = false) {
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;

    const auto nq0_pad = pad_nq(nq0);
    const auto nq1_pad = nq0_pad;
    const auto nq2_pad = nq0_pad;

    // Initialize the input and output arrays
    const auto size_JxW = nelmt * nq0 * nq1 * nq2;
    const auto size_inout = nelmt * nm0 * nm1 * nm2;

    auto JxW = make_constant_or_random_vec(size_JxW, T{1}, init_random);
    auto in = make_constant_or_random_vec(size_inout, T{3}, init_random);
    auto out = make_constant_vec(size_inout, T{0});
    auto basis0 = make_basis_vec<T>(nm0, nq0, nq0_pad);
    auto basis1 = make_basis_vec<T>(nm1, nq1, nq1_pad);
    auto basis2 = make_basis_vec<T>(nm2, nq2, nq2_pad);

    auto d_JxW = make_d_buf(JxW, queue);
    auto d_in = make_d_buf(in, queue);
    auto d_out = make_d_buf(out, queue);
    auto d_basis0 = make_d_buf(basis0, queue);
    auto d_basis1 = make_d_buf(basis1, queue);
    auto d_basis2 = make_d_buf(basis2, queue);

    // Performance in GDoF/s
    auto dof_rate = [&](double elapsed) {
        return 1.0e-9 * size_inout / elapsed;
    };
    auto byte_rate = [&](double elapsed) {
        return 1.0e-9 * sizeof(float) * (2 * size_inout + size_JxW) / elapsed;
    };

    auto params = kernel_params{.nq0 = nq0,
                                .nq0_pad = nq0_pad,
                                .block_size = 16,
                                .wgs = bk1_5_work_group_size(nq0)};
    auto bk1_kernel = make_kernel("src/bk1.ir", "bk1", params, queue);
    auto exe_range = tinytc::get_execution_range(
        bk1_kernel,
        sycl::range<3u>{1u, 1u,
                        static_cast<std::size_t>(nelmt / params.block_size)});
    if (nelmt % params.block_size != 0) {
        throw std::runtime_error("nelmt % block_size != 0");
    }

    double time_cl = std::numeric_limits<double>::max();
    for (unsigned int t = 0u; t < ntests; ++t) {
        auto start = high_resolution_clock::now();

        queue
            .submit([&](sycl::handler &h) {
                h.set_args(d_basis0.get(), d_basis1.get(), d_basis2.get(),
                           d_JxW.get(), nelmt, d_in.get(), nelmt, d_out.get(),
                           nelmt);
                h.parallel_for(exe_range, bk1_kernel);
            })
            .wait_and_throw();

        auto stop = high_resolution_clock::now();
        duration<double> elapsed = stop - start;
        time_cl = std::min(time_cl, elapsed.count());
    }

    std::cout << "# TinyTC -> nelmt " << nelmt
              << " GDoF/s = " << dof_rate(time_cl) << '\n';
    std::cout << "# TinyTC -> nelmt " << nelmt
              << " GB/s = " << byte_rate(time_cl) << '\n';

    if (show_norm) {
        // 2. Copy data from device buffer d_out to host buffer out
        queue.copy(d_out.get(), out.data(), size_inout).wait_and_throw();

        double normSqr{0.0};
        for (auto h : out) {
            normSqr += ((double)h) * ((double)h);
        }

        std::cout << "# SYCL kernel norm = " << std::sqrt(normSqr) << '\n';
    }
    if (show_norm) {
        queue.copy(d_out.get(), out.data(), out.size()).wait_and_throw();

        auto out_ref = std::vector<T>(out.size());
        bk1_reference(nelmt, params.block_size, nq0, nq1, nq2, nq0_pad, nq1_pad,
                      nq2_pad, basis0.data(), basis1.data(), basis2.data(),
                      JxW.data(), in.data(), out_ref.data());
        show_error_norm(out, out_ref);
    }
}

template <typename T>
void run_bk5(sycl::queue &queue, const int nq0, const int nq1, const int nq2,
             const std::int64_t nelmt, const std::int64_t ntests,
             bool show_norm, bool init_random = false) {
    const auto nq0_pad = pad_nq(nq0);
    const auto nq1_pad = nq0_pad;
    const auto nq2_pad = nq0_pad;

    const auto size_G = nelmt * nq0 * nq1 * nq2 * 6;
    const auto size_inout = nelmt * nq0 * nq1 * nq2;

    auto G = make_constant_or_random_vec(size_G, T{1}, init_random);
    auto in = make_constant_or_random_vec(size_inout, T{3}, init_random);
    auto out = make_constant_vec(size_inout, T{0});
    auto basis0 = make_basis_vec<T>(nq0, nq0, nq0_pad);
    auto basis1 = make_basis_vec<T>(nq1, nq1, nq1_pad);
    auto basis2 = make_basis_vec<T>(nq2, nq2, nq2_pad);

    auto d_G = make_d_buf(G, queue);
    auto d_in = make_d_buf(in, queue);
    auto d_out = make_d_buf(out, queue);
    auto d_basis0 = make_d_buf(basis0, queue);
    auto d_basis1 = make_d_buf(basis1, queue);
    auto d_basis2 = make_d_buf(basis2, queue);

    // Performance in GDoF/s
    auto dof_rate = [&](double elapsed) {
        return 1.0e-9 * nelmt * nq0 * nq1 * nq2 / elapsed;
    };
    auto byte_rate = [&](double elapsed) {
        return 1.0e-9 * sizeof(float) * (size_G + 2 * size_inout) / elapsed;
    };

    auto params = kernel_params{.nq0 = nq0,
                                .nq0_pad = nq0_pad,
                                .block_size = 16,
                                .wgs = bk1_5_work_group_size(nq0)};
    auto bk5_kernel = make_kernel("src/bk5.ir", "bk5", params, queue);
    auto exe_range = tinytc::get_execution_range(
        bk5_kernel,
        sycl::range<3u>{1u, 1u,
                        static_cast<std::size_t>(nelmt / params.block_size)});
    if (nelmt % params.block_size != 0) {
        throw std::runtime_error("nelmt % block_size != 0");
    }

    double time_cl = std::numeric_limits<double>::max();
    for (unsigned int t = 0u; t < ntests; ++t) {
        auto start = high_resolution_clock::now();

        queue
            .submit([&](sycl::handler &h) {
                h.set_args(d_basis0.get(), d_basis1.get(), d_basis2.get(),
                           d_G.get(), nelmt, d_in.get(), nelmt, d_out.get(),
                           nelmt);
                h.parallel_for(exe_range, bk5_kernel);
            })
            .wait_and_throw();

        auto stop = high_resolution_clock::now();
        duration<double> elapsed = stop - start;
        time_cl = std::min(time_cl, elapsed.count());
    }

    std::cout << "# TinyTC -> nelmt " << nelmt
              << " GDoF/s = " << dof_rate(time_cl) << '\n';
    std::cout << "# TinyTC -> nelmt " << nelmt
              << " GB/s = " << byte_rate(time_cl) << '\n';

    if (show_norm) {
        // 2. Copy data from device buffer d_out to host buffer out
        queue.copy(d_out.get(), out.data(), out.size()).wait_and_throw();

        auto out_ref = std::vector<T>(out.size());
        bk5_reference(nelmt, params.block_size, nq0, nq1, nq2, nq0_pad, nq1_pad,
                      nq2_pad, basis0.data(), basis1.data(), basis2.data(),
                      G.data(), in.data(), out_ref.data());
        show_error_norm(out, out_ref);
    }
}

int main(int argc, char **argv) {
    const unsigned int tc = argc > 1 ? atoi(argv[1]) : 1;
    const unsigned int nq0 = argc > 2 ? atoi(argv[2]) : 4;
    const unsigned int nq1 = nq0;
    const unsigned int nq2 = nq0;

    unsigned int nelmt = argc > 3 ? atoi(argv[3]) : 2 << 18;
    unsigned int ntests = argc > 4 ? atoi(argv[4]) : 5u;

    const char *env = getenv("SHOW_NORM");
    const bool show_norm = (env && strcmp(env, "1") == 0);

    const char *env_rnd = getenv("INIT_RANDOM");
    const bool init_random = (env_rnd && strcmp(env_rnd, "1") == 0);

    try {
        // This tries to create a queue on a GPU device if available,
        // otherwise it throws an exception.
        auto queue = sycl::queue(sycl::default_selector_v,
                                 sycl::property::queue::in_order());

        auto device = queue.get_device();

        std::cout << "# Test case: BK" << tc << ", nq0=" << nq0 << std::endl;
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

        switch (tc) {
            case 1:
                run_bk1<float>(queue, nq0, nq1, nq2, nelmt, ntests, show_norm,
                               init_random);
                break;
            case 5:
                run_bk5<float>(queue, nq0, nq1, nq2, nelmt, ntests, show_norm,
                               init_random);
                break;
            default:
                std::cerr << "Unknown test case: " << tc << std::endl;
                return 1;
        }

    } catch (tinytc::status const &st) {
        std::cerr << "tinytc error (" << static_cast<int>(st)
                  << "): " << to_string(st) << std::endl;
        return 1;
    } catch (sycl::exception const &e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    } catch (std::exception const &e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
