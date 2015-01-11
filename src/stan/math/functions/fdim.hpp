#ifndef STAN__MATH__FUNCTIONS__FDIM_HPP
#define STAN__MATH__FUNCTIONS__FDIM_HPP

#include <math.h> 
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {
    /** 
     * The positive difference function (C99).  
     *
     * The function is defined by
     *
     * <code>fdim(a,b) = (a > b) ? (a - b) : 0.0</code>.
     *
     * @param a First value.
     * @param b Second value.
     * @return Returns min(a - b, 0.0).
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1, T2>::type
    fdim(T1 a, T2 b) {
      if (boost::math::isnan(a) || boost::math::isnan(b))
        return std::numeric_limits<typename boost::math::tools::promote_args<T1, T2>::type>
          ::quiet_NaN();
      return ::fdim(a, b);
    } 
  }
}

#endif
