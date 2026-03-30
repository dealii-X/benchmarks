#include <iostream>
#include <kernels/BK1/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>


template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{
    const unsigned int nm = nq - 1;
    
    //Allocation of arrays
    T* basis = new T[nm * nq];
    T* JxW   = new T[nelmt * nq * nq * nq];
    T* in    = new T[nelmt * nm * nm * nm];
    T* out   = new T[nelmt * nm * nm * nm];


    //Initialize the input and output arrays
    std::fill(JxW, JxW + nelmt * nq * nq * nq, (T)1.0f);
    std::fill(in, in + nelmt * nm * nm * nm, (T)3.0f);
    std::fill(out, out + nelmt * nm * nm * nm, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq; p++)
    {
        for(unsigned int i = 0u; i < nm; i++)
        {
            basis[p * nm + i] = std::cos((T)(p * nm + i));
        }
    }

    BenchmarkPrinter<T> printer;

    // ------------------------- 2D Block(pq)---------------------------------------------------
    {
        std::vector<double> results = BK1::Parallel::Kokkos_MassOperator<T, nq>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, basis, JxW, in, out, ntests);
        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];

        uint64_t nDOF = (uint64_t)nm * nm * nm * nelmt; uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 *(2 * nDOF + nQuad) * sizeof(T) / time;
        printer("BK1", nm - 1, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }


    delete[] basis; delete[] JxW; delete[] in; delete[] out;
}


int main(int argc, char **argv){
    using T = float;
    int shmemPerBlock = 10800;

    Kokkos::initialize(argc, argv);

    BenchmarkPrinter<T> printer;
    printer.print_header();

    constexpr int    NumSample   = 50;
    constexpr double DOFmin      = 1e4;
    constexpr double DOFmax      = 1e8;

    const double log_step = std::pow(DOFmax / DOFmin, 1.0 / (NumSample - 1));

    for (int istep = 0; istep < NumSample; ++istep)
    {
        int dof = static_cast<int>(std::llround(DOFmin * std::pow(log_step, istep)));

         for (int nq = 3; nq <= 10; ++nq) {
            int nm = nq - 1;

            
            unsigned int nelmt = dof / (nm * nm * nm);
            if (nelmt == 0) continue;

            unsigned int nelmtPerBatch     = shmemPerBlock / (2 * nq * nq * nq) / sizeof(T);    if(nelmtPerBatch == 0) nelmtPerBatch = 1;
            unsigned int numBlocks         = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2;
            unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

            switch (nq) {
                case 3:  run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 4:  run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 5:  run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 6:  run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 7:  run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 8:  run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 9:  run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 10: run_test<T,10>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                default: break;
            }
        }
    }


    Kokkos::finalize();
    return 0;
}