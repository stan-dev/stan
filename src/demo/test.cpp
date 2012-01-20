#include <Eigen/Dense>

#include <iostream>
#include <vector>

#include <stan/agrad/agrad.hpp>

int main(int argc, char* argv[]) {

  using stan::agrad::var;

  var x1 = 2.0;
  var x2 = 3.0;

  std::cout << "sizeof(vari)=" << sizeof(stan::agrad::vari) << std::endl;

  std::vector<var> x;
  x.push_back(x1);
  x.push_back(x2);

  var fx = x1 * x2;
  
  std::vector<double> gx;
  fx.grad(x,gx);

  for (unsigned int i = 0; i < gx.size(); ++i)
    std::cout << "gx[" << i << "]=" << gx[i] << std::endl;

  var y1 = 4.0;
  var y2 = 5.0;
  
  std::vector<var> y;
  y.push_back(y1);
  y.push_back(y2);
  
  var fy = y1 + y2;
  
  std::vector<double> gy;
  fy.grad(y,gy);

  for (unsigned int i = 0; i < gy.size(); ++i)
    std::cout << "gy[" << i << "]=" << gy[i] << std::endl;
  
}
