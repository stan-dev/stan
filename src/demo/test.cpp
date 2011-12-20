#include <iostream>

#include <boost/exception/all.hpp>
#include <boost/math/policies/policy.hpp>

#include <stan/prob/distributions_error_handling.hpp>

int main() {
  double result = 0.0;
  try {
    stan::prob::check_bounds("bar_fun",
			     1.0, 0.0,
			     &result,
			     boost::math::policies::policy<>());
  } catch (std::exception e) {
    std::cout << "what=" << e.what() << std::endl;
    std::cout << "diagnostics=" << boost::diagnostic_information(e) << std::endl;
  }
}
