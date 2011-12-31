#include <iostream>
#include <vector>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>

int main() {
  using stan::agrad::var;
  using stan::agrad::vari;
  using stan::agrad::log_sum_exp;
  unsigned int I = 3;
  std::vector<var> vs(I);
  for (unsigned int i = 0; i < I; ++i)
    vs[i] = i;

  var lse = log_sum_exp(vs);

  std::cout << "sizeof(var)=" << sizeof(var) << std::endl;
  std::cout << "sizeof(vari)=" << sizeof(vari) << std::endl;
  std::cout << "sizeof(vector<vari*>)=" << sizeof(std::vector<vari*>) << std::endl;

  std::cout << "DONE" << std::endl;


}
