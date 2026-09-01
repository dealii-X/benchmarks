#include <iostream>
#include <cmath>
#include <algorithm>
#include <kernels/BK3/otf_templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(const size_t nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{   
    const unsigned int nm = nq - 1;
    const unsigned int nquad = nq * nq * nq;

    //Allocation of arrays
    T* basis = new T[nm * nq];
    T* dbasis = new T[nq * nq];
    T* weights = new T[nq];

    T* coord_q = new T[nelmt * 3 * nquad];
    T* in = new T[nelmt * nm*nm*nm];
    T* out = new T[nelmt * nm*nm*nm];


    //Initialize the input and output arrays
    for(size_t i = 0; i < nelmt * nm * nm * nm; ++i)
        in[i] = std::sin((T)i);

    std::fill(out, out + nelmt * nm*nm*nm, (T)0.0);
    std::fill(weights, weights + nq, (T)1.0);


    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
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

    //Initialize coord_q as the same stretched 3D grid used by the precomputed G benchmark
    for(size_t e = 0; e < nelmt; ++e){
        size_t coord_base = e * 3 * nquad;
        size_t xbase = coord_base;
        size_t ybase = coord_base + nquad;
        size_t zbase = coord_base + 2 * nquad;
        
        for(unsigned int p = 0; p < nq; ++p){
            for(unsigned int q = 0; q < nq; ++q){
                for(unsigned int r = 0; r < nq; ++r){
                    unsigned int idx = p * nq * nq + q * nq + r;
                    coord_q[xbase + idx] = (T)p + 0.1 * (T)q + 0.1 * (T)r;
                    coord_q[ybase + idx] = 0.1 * (T)p + (T)q + 0.1 * (T)r;
                    coord_q[zbase + idx] = 0.1 * (T)p + 0.1 * (T)q + (T)r;
                }
            }
        }
    }


    BenchmarkPrinter<T> printer;


    // ------------------------- OTF Kernel ---------------------------------------------------
    {
        std::vector<double> results = BK3::Parallel::Kokkos_LaplaceOperator_OTF<T, nq>(
            nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
            basis, dbasis, weights, coord_q, in, out, ntests);

        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];
        
        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt;
        uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 *(2 * nDOF + 3 * nQuad) * sizeof(T) / time;
        printer("BK3_OTF", nq - 2, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }


    delete[] basis; delete[] dbasis; delete[] weights; delete[] coord_q; delete[] in; delete[] out;
}

int main(int argc, char **argv){

    using T = double;
    int shmemPerBlock = 10000;

    Kokkos::initialize(argc, argv);


    BenchmarkPrinter<T> printer;
    printer.print_header();

    constexpr int NumSample = 50;
    constexpr double DOFmin = 1e4;
    constexpr double DOFmax = 1e8;

    const double log_step = std::pow(DOFmax / DOFmin, 1.0 / (NumSample - 1));

    for (int istep = 0; istep < NumSample; ++istep)
    {
        size_t dof = static_cast<size_t>(std::llround(DOFmin * std::pow(log_step, istep)));

        for (int nq = 3; nq <= 10; ++nq) {
            int nm = nq - 1;

            size_t nelmt = dof / (nm * nm * nm);
            if (nelmt == 0) continue;

            unsigned int nelmtPerBatch = std::max(1UL, shmemPerBlock / (4 * nq * nq * nq) / sizeof(T));
            unsigned int numBlocks = std::max((size_t)1, (nelmt + nelmtPerBatch - 1) / nelmtPerBatch);
            unsigned int threadsPerBlock = nq * nq * std::max(1u, nelmtPerBatch);

            switch (nq) {
                case 3: run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 4: run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 5: run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 6: run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 7: run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 8: run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 9: run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 10: run_test<T, 10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                default: break;
            }
        }
    }


    Kokkos::finalize();
    return 0;
}