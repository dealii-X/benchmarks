#include <iostream>
#include <cmath>
#include <math/lagrange_basis.hpp>

constexpr float x0_ =  -2.0f;
constexpr float x1_  =  0.5f;

constexpr float y0_  = -2.0f;
constexpr float y1_  =  0.0f;
constexpr float y2_  =  1.0f;

constexpr float z0_  = -3.0f;
constexpr float z1_  =  0.5f;
constexpr float z2_  =  1.0f;
constexpr float z3_  =  1.5f;


int main(){

  //partition of unity test
  std::vector<float> xcoordinates{x0_, x1_};
  std::vector<float> ycoordinates{y0_, y1_, y2_};
  std::vector<float> zcoordinates{z0_, z1_, z2_, z3_};

  LagrangeBasis<float> lbasis2(xcoordinates);
  LagrangeBasis<float> lbasis3(ycoordinates);
  LagrangeBasis<float> lbasis4(zcoordinates);
  
  float sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

  //partition of unity test
  for(float x =-2.0f; x <= 0.5f; x += 0.05)
  {
    sum2 += lbasis2(0,x) + lbasis2(1,x) - 1.0f;

    sum3 += lbasis3(0, x) + lbasis3(1, x) + lbasis3(2, x) - 1.0f;

    sum4 += lbasis4(0, x) + lbasis4(1, x) + lbasis4(2, x) + lbasis4(3, x) - 1.0f;
  }
  float eps = 1e-6;
  if(std::abs(sum2) < eps && std::abs(sum3) < eps && std::abs(sum4) < eps){
    std::cout << "Partiotion of unity test passed" << std::endl;
  } 
  else{
    std::cout << "Partiotion of unity test failed" << std::endl;
  }



  return 0;
}