#include <iostream>
#include <lagrange_basis.hpp>

int main(){

  //partition of unity test
  std::vector<float> xcoordinates{-2.0f, 0.5f};
  std::vector<float> ycoordinates{-2.0f, 0.0f, 1.0f};
  std::vector<float> zcoordinates{-3.0f, 0.5f, 1.0f, 1.5f};

  LagrangeBasis<float> lbasis2(xcoordinates);
  LagrangeBasis<float> lbasis3(ycoordinates);
  LagrangeBasis<float> lbasis4(zcoordinates);
  
  //partition of unity test
  for(float x=-2.0f; x <= 0.5f; x += 0.05)
  {
    //test for 2nd order
    std::cout << lbasis2(0, x) + lbasis2(1, x)  << " ";

    //test for 3rd order
    std::cout << lbasis3(0, x) + lbasis3(1, x) + lbasis3(2, x)  << " ";

    //test for 4rd order
    std::cout << lbasis4(0, x) + lbasis4(1, x) + lbasis4(2, x) + lbasis4(3, x) << " ";
    
  }
  std::cout << std::endl;

  //derivative test
  std::vector<float> coordinates {-0.5f, 0.5f};
  LagrangeBasis<float> lbasis1(coordinates);
  for(float x=-0.5f; x <= 0.5f; x += 0.05)
  {
      std::cout << lbasis1.derivative(0, x) << " ";
      std::cout << lbasis1.derivative(1, x) << " ";
  }
  std::cout << std::endl;
  return 0;
}