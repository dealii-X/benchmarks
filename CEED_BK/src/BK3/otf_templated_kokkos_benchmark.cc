#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <kernels/BK3/otf_templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(size_t nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;

    // Allocation of arrays
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];
    T* weights = new T[nq];

    T* coord_q = new T[nelmt * 3 * nquad];
    T* in = new T[nelmt * nm * nm * nm];
    T* out = new T[nelmt * nm * nm * nm];

    // Initialize the input array with varying data so the derivative is non-zero
    for(size_t i = 0; i < nelmt * nm * nm * nm; ++i){
        in[i] = std::sin((T)i); 
    }
    std::fill(out, out + nelmt * nm * nm * nm, (T)0.0);
    std::fill(weights, weights + nq, (T)1.0);

    // Initialize coord_q as a stretched 3D grid to ensure positive det(J)
    for(size_t e = 0; e < nelmt; ++e){
        size_t coord_base = e * 3 * nquad;
        size_t xbase = coord_base;
        size_t ybase = coord_base + nquad;
        size_t zbase = coord_base + 2 * nquad;
        
        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){
                    unsigned int idx = p * nq * nq + q * nq + r;
                    // x depends heavily on p, y on q, z on r
                    coord_q[xbase + idx] = (T)p + 0.1 * (T)q + 0.1 * (T)r;
                    coord_q[ybase + idx] = 0.1 * (T)p + (T)q + 0.1 * (T)r;
                    coord_q[zbase + idx] = 0.1 * (T)p + 0.1 * (T)q + (T)r;
                }
            }
        }
    }

    // Initialization of basis functions (varying data)
    for(unsigned int p = 0u; p < nq; p++){
        for(unsigned int i = 0u; i < nm; i++){
            basis[p * nm + i] = std::cos((T)(p * nm + i));
        }
    }

    //Initialization of dbasis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int p = 0u; p < nq; p++)
        {
            dbasis[i * nq + p] = std::cos((T)(i * nq + p));
        }
    }

    BenchmarkPrinter<T> printer;
    printer.print_header();

    // ------------------------- Kokkos OTF Kernel ---------------------------------------------------
    {
        std::vector<double> results = BK3::Parallel::Kokkos_LaplaceOperator_OTF<T, nq>(
            nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
            basis, dbasis, weights, coord_q, in, out, ntests);
            
        auto DOFs = results[0]; 
        auto sum = results[1]; 
        auto time = results[2];
        
        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt; 
        uint64_t nQuad = (uint64_t)nquad * nelmt;
        
        // BW formulation for OTF: Read IN(1), Write OUT(1), Read COORD(3) per element
        T bw = 1.0e-9 *(2 * nDOF + 3 * nQuad) * sizeof(T) / time;
        printer("BK3_OTF", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    delete[] basis; delete[] dbasis; delete[] weights; delete[] coord_q; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    Kokkos::initialize(argc, argv);

    unsigned int p                 = (argc > 1) ? atoi(argv[1]) : 2u;
    unsigned int nq                = p + 2;
    size_t nelmt                   = (argc > 2) ? std::stoull(argv[2]) : (1ULL << 16);
    
    unsigned int nelmtPerBatch     = (argc > 3) ? atoi(argv[3]) : std::max(1UL, shmemPerBlock / (4 * nq * nq * nq) / sizeof(T));
    unsigned int numBlocks         = (argc > 4) ? atoi(argv[4]) : std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);

    unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

    threadsPerBlock                = (argc > 5) ? atoi(argv[5]) : threadsPerBlock;
    unsigned int ntests            = (argc > 6) ? atoi(argv[6]) : 10u;

    std::cout.precision(8);

    switch(nq) {
        case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        case 10: run_test<T, 10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, ntests); break;
        default:
            std::cerr << "Error: Unsupported p value. Please use a value between 1 and 8." << std::endl;
            break;
    }

    Kokkos::finalize();
    return 0;
}