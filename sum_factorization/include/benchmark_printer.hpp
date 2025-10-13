#ifndef BENCHMARK_PRINTER_HPP
#define BENCHMARK_PRINTER_HPP

#include <iostream>
#include <string>
#include <iomanip>

class BenchmarkPrinter{
    public:
    BenchmarkPrinter(int precision = 3, int w_name = 15, int w_p0 = 4, int w_p1 = 4, int w_p2 = 4, int w_nelmt = 12, int w_numThreads = 16, int w_DOF = 16, int w_time = 10, int w_GDOFs = 8)
        :precision_{precision},
        widths_{
            w_name,
            w_p0,
            w_p1,
            w_p2,
            w_nelmt,
            w_numThreads,
            w_DOF,
            w_time,
            w_GDOFs
    }
    {
        std::cout << std::fixed << std::setprecision(precision_);
    }
    
    struct ColumnWidths
    {
        int name       ;
        int p0         ;
        int p1         ;
        int p2         ;
        int nelmt      ;
        int numThreads ;
        int DOF        ;
        int time       ;
        int GDOFs      ;
    };
    
    void print_header() const {
        std::cout << std::left  << std::setw(widths_.name)       << "Kernel"
                  << std::right << std::setw(widths_.p0)         << "p0"
                  << std::right << std::setw(widths_.p1)         << "p1"
                  << std::right << std::setw(widths_.p2)         << "p2"
                  << std::right << std::setw(widths_.nelmt)      << "nelmt"
                  << std::right << std::setw(widths_.numThreads) << "numThreads"
                  << std::right << std::setw(widths_.DOF)        << "DOF"
                  << std::right << std::setw(widths_.time)       << "time"
                  << std::right << std::setw(widths_.GDOFs)      << "GDOF/s"
        << std::endl;
    }
    void operator()(std::string name, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int nelmt, unsigned int numThreads, unsigned int DOF, double time, double GDOFs) const{
                        std::cout << std::left  << std::setw(widths_.name)       << name 
                                  << std::right << std::setw(widths_.p0)         << p0
                                  << std::right << std::setw(widths_.p1)         << p1
                                  << std::right << std::setw(widths_.p2)         << p2
                                  << std::right << std::setw(widths_.nelmt)      << nelmt
                                  << std::right << std::setw(widths_.numThreads) << numThreads
                                  << std::right << std::setw(widths_.DOF)        << DOF
                                  << std::right << std::setw(widths_.time)       << time
                                  << std::right << std::setw(widths_.GDOFs)      << GDOFs
                        << std::endl;
                    }
    
    private:
    int precision_;
    ColumnWidths widths_;

};

#endif  //BENCHMARK_PRINTER_HPP