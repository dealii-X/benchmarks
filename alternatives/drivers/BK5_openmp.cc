
#include <iostream>
#include <numeric>
#include <cmath>
#include <vector>
#include <cstring>

#include "timer.hpp"
#include "common.hpp"

template<typename T, int nq0, int nq1, int nq2, typename index_type = int>
void SumFactorization(
    const size_t nelmt, 
    const T *__restrict__ dbasis0, 
    const T *__restrict__ dbasis1,
    const T *__restrict__ dbasis2, 
    const T *__restrict__ G, 
    const T *__restrict__ in, 
          T *__restrict__ out)
{

    #pragma omp target \
        map(to: dbasis0[:nq0*nq0], dbasis1[:nq1*nq1], dbasis2[:nq2*nq2]) \
        map(to: G[:nelmt*6*nq0*nq1*nq2], in[:nelmt*nq0*nq1*nq2]) \
        map(from: out[:nelmt*nq0*nq1*nq2])
    #pragma omp teams loop
    for(size_t e = 0; e < nelmt; ++e){                    
        // Geometric vals
        T Grr, Grs, Grt, Gss, Gst, Gtt;

        // Intermediate vals
        T rqr[nq0 * nq1 * nq2];
        T rqs[nq0 * nq1 * nq2];
        T rqt[nq0 * nq1 * nq2];

        for(index_type i = 0; i < nq0; ++i){
            for(index_type j = 0; j < nq1; ++j){              
                for(index_type k = 0; k < nq2; ++k){

                    //Load Geometric Factors, coalesced access
                    Grr = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 0 * nq2 + k];
                    Grs = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 1 * nq2 + k];
                    Grt = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 2 * nq2 + k];
                    Gss = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 3 * nq2 + k];
                    Gst = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 4 * nq2 + k];
                    Gtt = G[e * nq0 * nq1 * 6 * nq2 + i * nq1 * 6 * nq2 + j * 6 * nq2 + 5 * nq2 + k];
                    
                    // Multiply by D
                    T qr = T(0); T qs = T(0); T qt = T(0);

                    for(index_type n = 0; n < nq0; ++n){
                        qr += in[e * nq0 * nq1 * nq2 + n * nq1 * nq2 + j * nq2 + k] * dbasis0[i * nq0 + n];
                    }

                    for(index_type n = 0; n < nq1; ++n){
                        qs += in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + n * nq2 + k] * dbasis1[j * nq1 + n];
                    }

                    for(index_type n = 0; n < nq2; ++n){
                        qt += in[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + n] * dbasis2[k * nq2 + n];
                    }

                    // Apply chain rule
                    rqr[i * nq1 * nq2 + j * nq2 + k] = Grr * qr + Grs * qs + Grt * qt;
                    rqs[i * nq1 * nq2 + j * nq2 + k] = Grs * qr + Gss * qs + Gst * qt;
                    rqt[i * nq1 * nq2 + j * nq2 + k] = Grt * qr + Gst * qs + Gtt * qt;
                }
            }
        }

        for(index_type i = 0; i < nq0; ++i){                      
            for(index_type j = 0; j < nq1; ++j){ 
                for(index_type k = 0; k < nq2; ++k){ 

                T tmp0 = T(0);
                for(index_type n = 0; n < nq0; ++n)
                    tmp0 += rqr[n * nq1 * nq2 + j * nq2 + k] * dbasis0[n * nq0 + i];

                for(index_type n = 0; n < nq1; ++n)                
                    tmp0 += rqs[i * nq1 * nq2 + n * nq2 + k] * dbasis1[n * nq1 + j];

                for(index_type n = 0; n < nq2; ++n)
                    tmp0 += rqt[i * nq1 * nq2 + j * nq2 + n] * dbasis2[n * nq2 + k];

                out[e * nq0 * nq1 * nq2 + i * nq1 * nq2 + j * nq2 + k] = tmp0;
                }
            }
        }
    }

}

template<typename T, int nq0, int nq1, int nq2>
void run_test(const size_t nelmt, const int ntests, bool show_norm = false)
{
    //Allocation of arrays
    std::vector<T> dbasis0(nq0 * nq0);
    std::vector<T> dbasis1(nq1 * nq1);
    std::vector<T> dbasis2(nq2 * nq2);
    std::vector<T> G(nelmt * nq0 * nq1 * nq2 * 6, T(2.0));
    std::vector<T> in(nelmt * nq0 * nq1 * nq2, T(3.0));
    std::vector<T> out(nelmt * nq0 * nq1 * nq2, T(0.0));

    //Initialization of basis functions
    for(unsigned int i = 0u; i < nq0; i++) {
        for(unsigned int a = 0u; a < nq0; a++) {
            dbasis0[i * nq0 + a] = std::cos((T)(i * nq0 + a));
        }
    }

    for(unsigned int j = 0u; j < nq1; j++) {
        for(unsigned int b = 0u; b < nq1; b++) {
            dbasis1[j * nq1 + b] = std::cos((T)(j * nq1 + b));
        }
    }
    
    for(unsigned int k = 0u; k < nq2; k++) {
        for(unsigned int c = 0u; c < nq2; c++) {
            dbasis2[k * nq2 + c] = std::cos((T)(k * nq2 + c));
        }
    }

    [[maybe_unused]] const size_t size_dbasis0 = dbasis0.size();
    [[maybe_unused]] const size_t size_dbasis1 = dbasis1.size();
    [[maybe_unused]] const size_t size_dbasis2 = dbasis2.size();
    [[maybe_unused]] const size_t size_G = G.size();
    [[maybe_unused]] const size_t size_in = in.size();
    [[maybe_unused]] const size_t size_out = out.size();

    T* d_dbasis0 = dbasis0.data();
    T* d_dbasis1 = dbasis1.data();
    T* d_dbasis2 = dbasis2.data();
    T* d_G = G.data();
    T* d_in = in.data();
    T* d_out = out.data();

    double time = std::numeric_limits<double>::max();
    Timer Timer;

    #pragma omp target data \
        map(to: d_dbasis0[:size_dbasis0], d_dbasis1[:size_dbasis1], d_dbasis2[:size_dbasis2]) \
        map(to: d_G[:size_G], d_in[:size_in]) \
        map(tofrom: d_out[:size_out])
    for (unsigned int t = 0u; t < ntests; ++t)
    {   
        Timer.start();

        SumFactorization<T, nq0, nq1, nq2>(
                nelmt,d_dbasis0, d_dbasis1, d_dbasis2, d_G, d_in, d_out);
        
        Timer.stop();
        time = std::min(time, Timer.elapsedSeconds());
    }

    std::cout << "SumFactorization -> " << "nelmt = " << nelmt << " GDoF/s = " << 1.0e-9 * nelmt * nq0 * nq1 * nq2 / time << std::endl; 

    if (show_norm) {
        double normSqr = squared_norm<T>(out.data(),out.size());
        std::cout << "# OpenMP kernel norm = " << std::sqrt(normSqr) << '\n';
    }

}


int main(int argc, char **argv){

    int nq0       = (argc > 1) ? atoi(argv[1]) : 4;
    int nq1       = (argc > 2) ? atoi(argv[2]) : nq0;
    int nq2       = (argc > 3) ? atoi(argv[3]) : nq0;
    size_t nelmt  = (argc > 4) ? atoi(argv[4]) : 2 << 18;
    int ntests    = (argc > 5) ? atoi(argv[5]) : 5;

    const char *env = getenv("SHOW_NORM");
    bool show_norm = (env && strcmp(env, "1") == 0);

    std::cout.precision(8);

// Note: adding more cases can increase the compilation time.

    if (nq0 == nq1 && nq1 == nq2) {
        switch (nq0) {
            case 2: run_test<float,2,2,2>(nelmt, ntests, show_norm); break;
            case 3: run_test<float,3,3,3>(nelmt, ntests, show_norm); break;
            case 4: run_test<float,4,4,4>(nelmt, ntests, show_norm); break;
            case 5: run_test<float,5,5,5>(nelmt, ntests, show_norm); break;
//            case 6: run_test<float,6,6,6>(nelmt, ntests, show_norm); break;
//            case 7: run_test<float,7,7,7>(nelmt, ntests, show_norm); break;
//            case 8: run_test<float,8,8,8>(nelmt, ntests, show_norm); break;
            default: return unsupported(nq0, nq1, nq2);
        }
    } else {
        // Mixed cases aren't supported
        return unsupported(nq0, nq1, nq2);
    }

    return 0;
}
