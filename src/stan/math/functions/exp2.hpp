#ifndef STAN__MATH__FUNCTIONS__EXP2_HPP
#define STAN__MATH__FUNCTIONS__EXP2_HPP

#include <boost/math/tools/promotion.hpp>
#include <math.h>
namespace stan {

  namespace math {

    /**
     * Return the exponent base 2 of the specified argument (C99).
     *
     * The exponent base 2 function is defined by
     *
     * <code>exp2(y) = pow(2.0,y)</code>.
     *
     * @param y Value.
     * @tparam T Type of scalar.
     * @return Exponent base 2 of value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    exp2(const T y) {
      using ::pow;
      return pow(2.0,y);
    }
  }
}

#endif
