#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <kernels/BK3/otf2_templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(size_t nelmt, const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;

    // 1. Allocate Kokkos Views instead of raw pointers
    Kokkos::View<T**,Kokkos::LayoutRight>       d_basis("basis", nm, nq);
    Kokkos::View<T**,Kokkos::LayoutRight>       d_dbasis("dbasis", nq, nq);
    Kokkos::View<T*>        d_weights("weights", nq);
    Kokkos::View<T*****>    d_coord("coord", nelmt, 3, nq, nq, nq);
    Kokkos::View<T****>     d_in("in", nelmt, nm, nm, nm);
    Kokkos::View<T****>     d_out("out", nelmt, nm, nm, nm);

    // 2. Initialize d_in with Kokkos::sin in parallel (mapping flat index to 4D)
    Kokkos::parallel_for("init_in", 
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {(int64_t)nelmt, (int64_t)nm, (int64_t)nm, (int64_t)nm}),
        KOKKOS_LAMBDA(int64_t e, int64_t i, int64_t j, int64_t k) {
            size_t flat_idx = ((e * nm + i) * nm + j) * nm + k;
            d_in(e, i, j, k) = Kokkos::sin(flat_idx);
        });

    // 3. Initialize d_out and d_weights using Kokkos::deep_copy
    Kokkos::deep_copy(d_out, (T)0.0);
    Kokkos::deep_copy(d_weights, (T)1.0);

    // 4. Initialize d_coord as a stretched 3D grid
    Kokkos::parallel_for("init_coord", 
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {(int64_t)nelmt, (int64_t)nq, (int64_t)nq, (int64_t)nq}),
        KOKKOS_LAMBDA(int64_t e, int64_t p, int64_t q, int64_t r) {
            d_coord(e, 0, p, q, r) = (T)p + 0.1 * (T)q + 0.1 * (T)r;
            d_coord(e, 1, p, q, r) = 0.1 * (T)p + (T)q + 0.1 * (T)r;
            d_coord(e, 2, p, q, r) = 0.1 * (T)p + 0.1 * (T)q + (T)r;
        });

    // 5. Initialize basis functions with Kokkos::cos
    Kokkos::parallel_for("init_basis", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {(int64_t)nm, (int64_t)nq}),
        KOKKOS_LAMBDA(int64_t i, int64_t p) {
            d_basis(i, p) = Kokkos::cos((i * nq + p));
        });

    // 6. Initialize dbasis functions with Kokkos::cos
    Kokkos::parallel_for("init_dbasis", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {(int64_t)nq, (int64_t)nq}),
        KOKKOS_LAMBDA(int64_t i, int64_t p) {
            d_dbasis(i, p) = Kokkos::cos((i * nq + p));
        });

    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- Kokkos OTF2 Kernel ---------------------------------------------------
    {
        std::vector<double> results = BK3::Parallel::Kokkos_LaplaceOperator_OTF2<T, nq>(
            nelmt, numBlocks, threadsPerBlock, 
            d_basis, d_dbasis, d_weights, d_coord, d_in, d_out, ntests);
            
        auto DOFs = results[0]; 
        auto sum = results[1]; 
        auto time = results[2];
        
        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt; 
        uint64_t nQuad = (uint64_t)nquad * nelmt;
        
        T bw = 1.0e-9 * (2 * nDOF + 3 * nQuad) * sizeof(T) / time;
        printer("BK3_OTF2", nq - 2, nelmt, threadsPerBlock, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }
}

int main(int argc, char **argv){
    using T = double;


    Kokkos::initialize(argc, argv);

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u; unsigned int nq = p + 2;
    size_t nelmt                   = (argc > 2) ? std::stoull(argv[2]) : (1ULL << 16);
    unsigned int threadsPerBlock   = (argc > 3) ? atoi(argv[3]) : 32;    //or nelmntPerBlock
    const size_t nelmt_padded      = ((nelmt + threadsPerBlock - 1) / threadsPerBlock) * threadsPerBlock;
    
    unsigned int numBlocks         = (argc > 4) ? atoi(argv[4]) : (nelmt_padded + threadsPerBlock - 1) / threadsPerBlock;
    unsigned int ntests            = (argc > 5) ? atoi(argv[5]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 3: run_test<T, 3>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        case 10: run_test<T, 10>(nelmt_padded, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    Kokkos::finalize();
    return 0;
}