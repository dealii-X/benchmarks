#include <iostream>
#include <serial_kernels.hpp>
#include <cmath>

int main(int argc, char **argv){
    unsigned int nq0                = (argc > 1) ? atoi(argv[1]) : 4u;
    unsigned int nq1                = (argc > 2) ? atoi(argv[2]) : 4u;
    unsigned int nq2                = (argc > 3) ? atoi(argv[3]) : 4u;
    unsigned int nelmt              = (argc > 6) ? atoi(argv[6]) : 2 << 18;
    unsigned int ntests             = (argc > 6) ? atoi(argv[7]) : 10u;


    const unsigned int nm0 = nq0 - 1;
    const unsigned int nm1 = nq1 - 1;
    const unsigned int nm2 = nq2 - 1;

    //Allocation of arrays
    double* basis0 = new double[nm0 * nq0];
    double* basis1 = new double[nm1 * nq1];
    double* basis2 = new double[nm2 * nq2];
    double* in = new double[nelmt * nm0 * nm1 * nm2];
    double* out = new double[nelmt * nq0 * nq1 * nq2];

    //Initialize the input and output arrays
    std::fill(in, in + nelmt * nm0 * nm1 * nm2, (double)3);
    std::fill(out, out + nelmt * nq0 * nq1 * nq2, (double)0);

    //Initialization of basis functions
    for(unsigned int p = 0u; p < nq0; p++)
    {
        for(unsigned int i = 0u; i < nm0; i++)
        {
            basis0[p * nm0 + i] = std::cos((double)(p * nm0 + i));
        }
    }
    for(unsigned int q = 0u; q < nq1; q++)
    {
        for(unsigned int j = 0u; j < nm1; j++)
        {
            basis1[q * nm1 + j] = std::cos((double)(q * nm1 + j));
        }
    }
    for(unsigned int r = 0u; r < nq2; r++)
    {
        for(unsigned int k = 0u; k < nm2; k++)
        {
            basis2[r * nm2 + k] = std::cos((double)(r * nm2 + k));
        }
    }


    //Serial Kernels
    float SerialDirectEval = Serial::DirectEvaluation<double>(nm0, nm1, nm2, nq0, nq1, nq2, nelmt, basis0, basis1, basis2, in, out);
    float SerialSumFact = Serial::SumFactorization<double>(nm0, nm1, nm2, nq0, nq1, nq2, nelmt, basis0, basis1, basis2, in, out);

    std::cout << "SerialDirectEval<double> norm = " << std::sqrt(SerialDirectEval) << "\n";
    std::cout << "SerialSumFact<double> norm = " << std::sqrt(SerialSumFact) << "\n";
    return 0;
}