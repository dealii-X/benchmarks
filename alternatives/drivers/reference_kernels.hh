// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef REFERENCE_KERNELS_20251014_HH
#define REFERENCE_KERNELS_20251014_HH

#include <array>
#include <utility>
#include <vector>

template <typename IdxT, std::size_t Dim>
auto default_col_major(std::array<IdxT, Dim> const &shape) {
    auto stride = std::array<IdxT, Dim>{};
    stride[0] = 1;
    for (std::size_t d = 0; d < Dim - 1; ++d) {
        stride[d + 1] = stride[d] * shape[d];
    }
    return stride;
}
template <typename IdxT, std::size_t Dim>
auto default_row_major(std::array<IdxT, Dim> const &shape) {
    auto stride = std::array<IdxT, Dim>{};
    stride[Dim - 1] = 1;
    for (std::size_t d = Dim - 1; d > 0; --d) {
        stride[d - 1] = stride[d] * shape[d];
    }
    return stride;
}

template <std::size_t Dim, typename IdxT = int>
class tensor_layout {
   public:
    tensor_layout(std::array<IdxT, Dim> const &shape)
        : tensor_layout(shape, default_row_major(shape)) {}
    tensor_layout(std::array<IdxT, Dim> const &shape,
                  std::array<IdxT, Dim> const &stride)
        : shape_{shape}, stride_{stride} {}

    auto shape(std::size_t i) const { return shape_[i]; }
    auto stride(std::size_t i) const { return stride_[i]; }
    auto size() const {
        std::size_t max_sh = 0;
        std::size_t max_st = 0;
        for (std::size_t d = 0; d < Dim; ++d) {
            if (stride_[d] > max_st ||
                (stride_[d] == max_st && shape_[d] > max_sh)) {
                max_sh = shape_[d];
                max_st = stride_[d];
            }
        }
        return max_sh * max_st;
    }

    template <typename... I>
    auto operator()(I &&...indices) const -> std::size_t {
        static_assert(sizeof...(I) == Dim,
                      "Number of indices must match tensor dimension");
        return linear_index(std::array<IdxT, Dim>{
            static_cast<IdxT>(std::forward<I>(indices))...});
    }
    auto linear_index(std::array<IdxT, Dim> const &idx) const -> std::size_t {
        std::size_t l = 0;
        for (std::size_t d = 0; d < Dim; ++d) {
            l += idx[d] * stride_[d];
        }
        return l;
    }

   private:
    std::array<IdxT, Dim> shape_, stride_;
};

template <typename T, std::size_t Dim, typename IdxT = int>
class tensor_view {
   public:
    tensor_view(T *data, std::array<IdxT, Dim> const &shape)
        : data_{data}, layout_{shape} {}
    tensor_view(T *data, std::array<IdxT, Dim> const &shape,
                std::array<IdxT, Dim> const &stride)
        : data_{data}, layout_{shape, stride} {}
    tensor_view(T *data, tensor_layout<Dim> const &layout)
        : data_{data}, layout_{layout} {}

    auto layout() const -> tensor_layout<Dim> const & { return layout_; }

    template <typename... I>
    auto operator()(I &&...indices) -> T & {
        return data_[layout_(std::forward<I>(indices)...)];
    }
    template <typename... I>
    auto operator()(I &&...indices) const -> T const & {
        return data_[layout_(std::forward<I>(indices)...)];
    }

    auto get() -> T * { return data_; }
    void set(T *data) { data_ = data; }

   private:
    T *data_;
    tensor_layout<Dim, IdxT> layout_;
};

template <typename T, std::size_t Dim, typename IdxT = int>
class tensor : public tensor_view<T, Dim, IdxT> {
   public:
    using base = tensor_view<T, Dim, IdxT>;
    tensor(std::array<IdxT, Dim> const &shape) : base{nullptr, shape} {
        init();
    }
    tensor(std::array<IdxT, Dim> const &shape,
           std::array<IdxT, Dim> const &stride)
        : base{nullptr, shape, stride} {
        init();
    }
    tensor(tensor_layout<Dim> const &layout) : base{nullptr, layout} { init(); }

   private:
    void init() {
        managed_data_.resize(base::layout().size());
        base::set(managed_data_.data());
    }
    std::vector<T> managed_data_;
};

template <typename T>
auto bk1_reference(int nelmt, int block_size, int nq0, int nq1, int nq2,
                   int nq0_pad, int nq1_pad, int nq2_pad, T *basis0_p,
                   T *basis1_p, T *basis2_p, T *JxW_p, T *in_p, T *out_p) {
    const int nm0 = nq0 - 1;
    const int nm1 = nq1 - 1;
    const int nm2 = nq2 - 1;
    auto nblocks = 1 + (nelmt - 1) / block_size;
    auto basis0 = tensor_view<T, 2>(basis0_p, {nq0, nm0}, {nq0_pad, 1});
    auto basis1 = tensor_view<T, 2>(basis1_p, {nq1, nm1}, {nq1_pad, 1});
    auto basis2 = tensor_view<T, 2>(basis2_p, {nq2, nm2}, {nq2_pad, 1});
    auto JxW = tensor_view<T, 5>(JxW_p, {nblocks, nq2, nq1, nq0, block_size});
    auto in = tensor_view<T, 5>(in_p, {nblocks, nm2, nm1, nm0, block_size});
    auto out = tensor_view<T, 5>(out_p, {nblocks, nm2, nm1, nm0, block_size});

    auto wsp1 = tensor<T, 3>({nq0, nm1, nm2});
    auto wsp2 = tensor<T, 3>({nq0, nq1, nm2});
    auto wsp5 = tensor<T, 3>({nq0, nq1, nm2});
    auto wsp6 = tensor<T, 3>({nq0, nm1, nm2});

    for (int block = 0; block < nblocks; ++block) {
        for (int el = 0; el < block_size; ++el) {
            for (int a = 0; a < nq0; ++a) {
                for (int j = 0; j < nm1; ++j) {
                    for (int k = 0; k < nm2; ++k) {
                        auto tmp = T{0};
                        for (int i = 0; i < nm0; ++i) {
                            tmp += in(block, i, j, k, el) * basis0(a, i);
                        }
                        wsp1(a, j, k) = tmp;
                    }
                }
            }
            for (int a = 0; a < nq0; ++a) {
                for (int b = 0; b < nq1; ++b) {
                    for (int k = 0; k < nm2; ++k) {
                        auto tmp = T{0};
                        for (int j = 0; j < nm1; ++j) {
                            tmp += wsp1(a, j, k) * basis1(b, j);
                        }
                        wsp2(a, b, k) = tmp;
                    }
                }
            }
            for (int a = 0; a < nq0; ++a) {
                for (int b = 0; b < nq1; ++b) {
                    for (int k = 0; k < nm2; ++k) {
                        wsp5(a, b, k) = 0.0;
                    }
                    for (int c = 0; c < nq2; ++c) {
                        auto wsp3 = T{0};
                        for (int k = 0; k < nm2; ++k) {
                            wsp3 += wsp2(a, b, k) * basis2(c, k);
                        }
                        auto wsp4 = wsp3 * JxW(block, a, b, c, el);
                        for (int k = 0; k < nm2; ++k) {
                            wsp5(a, b, k) += wsp4 * basis2(c, k);
                        }
                    }
                }
            }
            for (int a = 0; a < nq0; ++a) {
                for (int j = 0; j < nm1; ++j) {
                    for (int k = 0; k < nm2; ++k) {
                        auto tmp = T{0};
                        for (int b = 0; b < nq1; ++b) {
                            tmp += wsp5(a, b, k) * basis1(b, j);
                        }
                        wsp6(a, j, k) = tmp;
                    }
                }
            }
            for (int i = 0; i < nm0; ++i) {
                for (int j = 0; j < nm1; ++j) {
                    for (int k = 0; k < nm2; ++k) {
                        auto tmp = T{0};
                        for (int a = 0; a < nq0; ++a) {
                            tmp += wsp6(a, j, k) * basis0(a, i);
                        }
                        out(block, i, j, k, el) = tmp;
                    }
                }
            }
        }
    }
}

template <typename T>
auto bk5_reference(int nelmt, int block_size, int nq0, int nq1, int nq2,
                   int nq0_pad, int nq1_pad, int nq2_pad, T *dbasis0_p,
                   T *dbasis1_p, T *dbasis2_p, T *G_p, T *in_p, T *out_p) {
    auto nblocks = 1 + (nelmt - 1) / block_size;
    auto dbasis0 = tensor_view<T, 2>(dbasis0_p, {nq0, nq0}, {nq0_pad, 1});
    auto dbasis1 = tensor_view<T, 2>(dbasis1_p, {nq1, nq1}, {nq1_pad, 1});
    auto dbasis2 = tensor_view<T, 2>(dbasis2_p, {nq2, nq2}, {nq2_pad, 1});
    auto G = tensor_view<T, 6>(G_p, {nblocks, nq2, nq1, 6, nq0, block_size});
    auto in = tensor_view<T, 5>(in_p, {nblocks, nq2, nq1, nq0, block_size});
    auto out = tensor_view<T, 5>(out_p, {nblocks, nq2, nq1, nq0, block_size});

    auto rqr = tensor<T, 3>({nq0, nq1, nq2});
    auto rqs = tensor<T, 3>({nq0, nq1, nq2});
    auto rqt = tensor<T, 3>({nq0, nq1, nq2});

    for (int b = 0; b < nblocks; ++b) {
        for (int e = 0; e < block_size; ++e) {
            for (int i = 0; i < nq0; ++i) {
                for (int j = 0; j < nq1; ++j) {
                    for (int k = 0; k < nq2; ++k) {
                        T qr = 0.0;
                        T qs = 0.0;
                        T qt = 0.0;

                        for (int n = 0; n < nq0; ++n) {
                            qr += in(b, n, j, k, e) * dbasis0(i, n);
                        }

                        for (int n = 0; n < nq1; ++n) {
                            qs += in(b, i, n, k, e) * dbasis1(j, n);
                        }

                        for (int n = 0; n < nq2; ++n) {
                            qt += in(b, i, j, n, e) * dbasis2(k, n);
                        }
                        auto G1 = G(b, i, j, 0, k, e);
                        auto G2 = G(b, i, j, 1, k, e);
                        auto G3 = G(b, i, j, 2, k, e);
                        auto G4 = G(b, i, j, 3, k, e);
                        auto G5 = G(b, i, j, 4, k, e);
                        auto G6 = G(b, i, j, 5, k, e);
                        rqr(i, j, k) = G1 * qr + G2 * qs + G3 * qt;
                        rqs(i, j, k) = G2 * qr + G4 * qs + G5 * qt;
                        rqt(i, j, k) = G3 * qr + G5 * qs + G6 * qt;
                    }
                }
            }

            for (int i = 0; i < nq0; ++i) {
                for (int j = 0; j < nq1; ++j) {
                    for (int k = 0; k < nq2; ++k) {
                        T tmp0 = T{0};
                        for (int n = 0; n < nq2; ++n) {
                            tmp0 += rqt(i, j, n) * dbasis2(n, k);
                        }
                        for (int n = 0; n < nq1; ++n) {
                            tmp0 += rqs(i, n, k) * dbasis1(n, j);
                        }
                        for (int n = 0; n < nq0; ++n) {
                            tmp0 += rqr(n, j, k) * dbasis0(n, i);
                        }

                        out(b, i, j, k, e) = tmp0;
                    }
                }
            }
        }
    }
}

#endif  // REFERENCE_KERNELS_20251014_HH
