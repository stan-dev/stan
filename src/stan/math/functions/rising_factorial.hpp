#ifndef __STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP__
#define __STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP__

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    template<typename T>
    inline T
    rising_factorial(const T x, const int n) { 
      using boost::math::tgamma;
      return tgamma(x + n) / tgamma(x); 
    }

  }
}

#endif
