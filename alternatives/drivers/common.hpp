#ifndef COMMON_HPP
#define COMMON_HPP

#include <algorithm>

template<typename T, typename AccumT = double>
AccumT squared_norm(const T* data, std::size_t size) {
    return std::transform_reduce(
        data, data + size,
        data,
        AccumT{},
        [](AccumT lhs, AccumT rhs) {
            return lhs + rhs;
        },
        [](T a, T b) {
            return static_cast<AccumT>(a) * static_cast<AccumT>(b);
        }
    );
}

// Helper function to reject unsupported combinations
static int unsupported(int nq0, int nq1, int nq2) {
    std::cerr << "unsupported combination: (" << nq0 << "," << nq1 << "," << nq2 << ")\n";
    return 1;
}

#endif // COMMON_HPP