#include <iostream>
#include <boost/math/distributions/detail/common_error_handling.hpp>
int main(int argc, char* argv[]) {
  std::cout << "isfinite(0U)=" << (boost::math::isfinite)(0U) << std::endl;
  std::cout << "isfinite(1U)=" << (boost::math::isfinite)(0U) << std::endl;
}
