#include <iostream>
#include <vector>

#include <stan/agrad/agrad.hpp>
#include <stan/agrad/error_handling.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/maths/matrix.hpp>

int main(int argc, char* argv[]) {

  using stan::agrad::var;
  using stan::agrad::print_stack;
  using stan::maths::check_not_nan;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  const char* f = "fun";
  const char* name = "x";

  var x = 1.0;

  var result;
  check_not_nan(f,x,name,&result,
                boost::math::policies::policy<>());
  print_stack(std::cout);

  std::cout << "DONE." << std::endl;
}
