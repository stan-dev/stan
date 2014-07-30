#ifndef STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP
#define STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP

#include <stan/math/functions/log_rising_factorial.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    rising_factorial(const T1 x, const T2 n) { 
      return std::exp(stan::math::log_rising_factorial(x,n)); 
    }

  }
}

#endif
