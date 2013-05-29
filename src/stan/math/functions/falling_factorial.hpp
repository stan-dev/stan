#ifndef __STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP__
#define __STAN__MATH__FUNCTIONS__FALLING_FACTORIAL_HPP__

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type 
    falling_factorial(const T1 x, const T2 n) { 
      using boost::math::tgamma;
      return tgamma(x + 1) / tgamma(n + 1); 
    }

  }
}

#endif
