#ifndef __STAN__MATH__FUNCTIONS__LOG1P_HPP__
#define __STAN__MATH__FUNCTIONS__LOG1P_HPP__

#include <limits>
#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Return the natural logarithm of one plus the specified value.
     *
     * The main use of this function is to cut down on intermediate
     * values during algorithmic differentiation.
     *
     * @param x Specified value.
     * @return Natural log of one plus <code>x</code>.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1p(const T x) {
      using std::log;
      if (!(x >= -1.0))
        return std::numeric_limits<double>::quiet_NaN();

      if (x > 1e-9 || x < -1e-9)
        return log(1.0 + x);     // direct, if distant from 1
      else if (x > 1e-16 || x < -1e-16)
        return x - 0.5 * x * x;  // 2nd order Taylor, if close to 1
      else
        return x;                // 1st order Taylor, if very close to 1
    }

  }
}

#endif
