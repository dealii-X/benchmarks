#include <iostream>
#include <kernels/BK5/templated_kokkos_kernels.hpp>
#include <timer.hpp>
#include <benchmark_printer.hpp>

template<typename T, const unsigned int nq>
void run_test(const unsigned int nelmt, const unsigned int nelmtPerBatch, 
    const unsigned int numBlocks, const unsigned int threadsPerBlock, const unsigned int ntests)
{
    //Allocation of arrays
    T* dbasis = new T[nq * nq];
    T* G = new T[nelmt * nq * nq * nq * 6];
    T* in = new T[nelmt * nq * nq * nq];
    T* out = new T[nelmt * nq * nq * nq];


    //Initialize the input and output arrays
    std::fill(G, G + nelmt * 6 * nq * nq * nq, (T)2.0f);
    std::fill(in, in + nelmt * nq * nq * nq, (T)3.0f);
    std::fill(out, out + nelmt * nq * nq * nq, (T)0.0f);


    //Initialization of basis functions
    for(unsigned int i = 0u; i < nq; i++)
    {
        for(unsigned int a = 0u; a < nq; a++)
        {
            dbasis[i * nq + a] = std::cos((T)(i * nq + a));
        }
    }


    BenchmarkPrinter<T> printer;


    {
        std::vector<double> results = BK5::Parallel::Kokkos_LaplaceOperator<T, nq>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 
                                            dbasis, G, in, out, ntests);
        auto DOFs = results[0]; auto sum = results[1]; auto time = results[2];
        
        uint64_t nDOF = (uint64_t)nq * nq * nq * nelmt; uint64_t nQuad = (uint64_t)nq * nq * nq * nelmt;
        T bw = 1.0e-9 *(2 * nDOF + 6*nQuad) * sizeof(T) / time;
        printer("BK5", nq - 1, nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, nDOF, time, DOFs, bw, std::sqrt(sum));
    }

    
    delete[] dbasis; delete[] G; delete[] in; delete[] out;
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

         for (int nq = 2; nq <= 9; ++nq) {
            int nm = nq;
            
            unsigned int nelmt = dof / (nm * nm * nm);
            if (nelmt == 0) continue;

                unsigned int nelmtPerBatch     = shmemPerBlock / (4 * nq * nq * nq) / sizeof(T);    if(nelmtPerBatch == 0) nelmtPerBatch = 1;
                unsigned int numBlocks         = (nelmt + nelmtPerBatch - 1) / nelmtPerBatch / 2;   
                unsigned int threadsPerBlock   = nq * nq * std::max(1u, nelmtPerBatch);

            switch (nq) {
                case 2:  run_test<T, 2>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 3:  run_test<T, 3>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 4:  run_test<T, 4>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 5:  run_test<T, 5>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 6:  run_test<T, 6>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 7:  run_test<T, 7>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 8:  run_test<T, 8>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                case 9:  run_test<T, 9>(nelmt, nelmtPerBatch, numBlocks, threadsPerBlock, 10); break;
                default: break;
            }
        }
    }


    Kokkos::finalize();
    return 0;
}