#ifndef STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP
#define STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP

#include <stan/math/functions/log_falling_factorial.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    falling_factorial(const T1 x, const T2 n) { 
      return std::exp(stan::math::log_falling_factorial(x,n)); 
    }

  }
}

#endif
