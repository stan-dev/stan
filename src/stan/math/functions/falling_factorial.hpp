#ifndef __STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP__
#define __STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP__

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    template<typename T>
    inline T
    falling_factorial(const T x, const int n) { 
      using boost::math::lgamma;
      return std::exp(lgamma(x + 1) - lgamma(n + 1)); 
    }

  }
}

#endif
