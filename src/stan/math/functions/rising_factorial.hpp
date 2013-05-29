#ifndef __STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP__
#define __STAN__MATH__FUNCTIONS__RISING_FACTORIAL_HPP__

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type 
    rising_factorial(const T1 x, const T2 n) { 
      using boost::math::tgamma;
      return tgamma(x + n) / tgamma(x); 
    }

  }
}

#endif
