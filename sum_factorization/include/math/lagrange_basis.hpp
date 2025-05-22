#ifndef LAGRANGE_BASIS_HPP
#define LAGRANGE_BASIS_HPP

#include <cassert>
#include <vector>
#include <cmath>

template<typename T>
struct LagrangeBasis{
  LagrangeBasis(const std::vector<T>& coordinates): coordinates_(coordinates)
  {
    assert(!coordinates_.empty());
    denominators_.resize(coordinates_.size());

    T denominator;
    for(std::size_t j = 0; j < denominators_.size(); ++j)
    {
      denominator = 1;
      for(std::size_t m = 0; m < coordinates_.size(); ++m)
      {
        if(j!=m){
          denominator *= coordinates_[j] - coordinates_[m];
        }
      }
      denominators_[j] = denominator;
    }
  }
  
  T operator()(const unsigned int j, T coordinate) const
  {
    assert(j < coordinates_.size());
    
    T result = 1;
    for(std::size_t m = 0; m < coordinates_.size(); m++)
    {
      if(m != j){
        result *= (coordinate - coordinates_[m]);
      }
    }
    result = result / denominators_[j];
    return result;
  }

  T derivative(const std::size_t j, T coordinate) const
  {
    assert(j < coordinates_.size());
    T Sum = 0;
    T product;
    for(std::size_t i = 0; i < coordinates_.size(); ++i)
    {
      product = 1;
      if(i != j)
      {
        for(std::size_t m = 0; m < coordinates_.size(); ++m)
        {
          if(m != i && m != j)
          {
            product *= (coordinate - coordinates_[m]) / (coordinates_[j] - coordinates_[m]);
          }
        }
        Sum += (1 / (coordinates_[j] - coordinates_[i])) * product;
      }
    }
    return Sum;
  }
  
  private:
  std::vector<T> coordinates_;
  std::vector<T> denominators_;
};

#endif  //LAGRANGE_BASIS_HPP