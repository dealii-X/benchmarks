#ifndef LOGSPACE_HPP
#define LOGSPACE_HPP

#include <vector>
#include <cmath>

template <typename T>
std::vector<T> logspace(T begin_exp, T end_exp, unsigned int numPoints, T base = 10){
    std::vector<T> points;
    points.reserve(numPoints);

    T step = (end_exp - begin_exp) / (numPoints - 1);
    T exponent;
    for(unsigned int i = 0; i < numPoints; ++i){
        exponent = begin_exp + i * step;
        points.emplace_back(std::pow(base, exponent));
    }

    return points;
}


#endif  //LOGSPACE_HPP














