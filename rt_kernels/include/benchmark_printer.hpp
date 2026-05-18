#ifndef BENCHMARK_PRINTER_HPP
#define BENCHMARK_PRINTER_HPP

#include <iostream>
#include <string>
#include <iomanip>

template<typename T>
class BenchmarkPrinter{
    public:
    BenchmarkPrinter(int precision = 3, int w_name = 15, int w_p = 4,
                     int w_nelmt = 12, int w_nelmtPerBatch = 14, int w_numBlocks = 12, int w_threadsPerBlock = 18, int w_DOF = 16,
                     int w_time = 12, int w_GDOFs = 12, int w_bw = 12, int w_check = 12)
        :precision_{precision},
        widths_{
            w_name,
            w_p,
            w_nelmt,
            w_nelmtPerBatch,
            w_numBlocks,
            w_threadsPerBlock,
            w_DOF,
            w_time,
            w_GDOFs,
            w_bw,
            w_check,
    }
    {
        std::cout << std::scientific << std::setprecision(precision_);
    }
    
    struct ColumnWidths
    {
        int name            ;
        int p               ;
        int nelmt           ;
        int nelmtPerBatch   ;
        int numBlocks       ;
        int threadsPerBlock ;
        int DOF             ;
        int time            ;
        int GDOFs           ;
        int bw              ;
        int check           ;
    };
    
    void print_header() const {
        std::cout << std::left  << std::setw(widths_.name)            << "Kernel"
                  << std::right << std::setw(widths_.p)               << "p"
                  << std::right << std::setw(widths_.nelmt)           << "nelmt"
                  << std::right << std::setw(widths_.nelmtPerBatch)   << "nelmtPerBatch"
                  << std::right << std::setw(widths_.numBlocks)       << "numBlocks"
                  << std::right << std::setw(widths_.threadsPerBlock) << "threadsPerBlock"
                  << std::right << std::setw(widths_.DOF)             << "DOF"
                  << std::right << std::setw(widths_.time)            << "time"
                  << std::right << std::setw(widths_.GDOFs)           << "GDOF/s"
                  << std::right << std::setw(widths_.bw)              << "bw(GB/s)"
                  << std::right << std::setw(widths_.check)           << "check"
        << std::endl;
    }

    void operator()(std::string name, int p, int nelmt, int nelmtPerBatch, int numBlocks, int threadsPerBlock, int DOF, T time, T GDOFs, T bw, T check) const{
                        std::cout << std::left  << std::setw(widths_.name)       << name 
                                  << std::right << std::setw(widths_.p)          << p
                                  << std::right << std::setw(widths_.nelmt)      << nelmt
                                  << std::right << std::setw(widths_.nelmtPerBatch)      << nelmtPerBatch
                                  << std::right << std::setw(widths_.numBlocks)  << numBlocks
                                  << std::right << std::setw(widths_.threadsPerBlock)  << threadsPerBlock
                                  << std::right << std::setw(widths_.DOF)        << DOF
                                  << std::right << std::setw(widths_.time)       << time
                                  << std::right << std::setw(widths_.GDOFs)      << GDOFs
                                  << std::right << std::setw(widths_.bw)         << bw
                                  << std::right << std::setw(widths_.check)      << check
                        << std::endl;
                    }
    
    private:
    int precision_;
    ColumnWidths widths_;

};

#endif  //BENCHMARK_PRINTER_HPP