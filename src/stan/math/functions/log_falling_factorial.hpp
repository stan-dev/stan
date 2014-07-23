#ifndef STAN__MATH__FUNCTIONS__LOG_FALLING_FACTORIAL_HPP
#define STAN__MATH__FUNCTIONS__LOG_FALLING_FACTORIAL_HPP

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    log_falling_factorial(const T1 x, const T2 n) { 
      using boost::math::lgamma;
      return lgamma(x + 1) - lgamma(n + 1); 
    }

  }
}

#endif
