#ifndef STAN_MATH_PRIM_SCAL_FUN_LOG1P_HPP
#define STAN_MATH_PRIM_SCAL_FUN_LOG1P_HPP

#include <boost/math/tools/promotion.hpp>
#include <limits>

namespace stan {
  namespace math {

    /**
     * Return the natural logarithm of one plus the specified value.
     *
     * The main use of this function is to cut down on intermediate
     * values during algorithmic differentiation.
     *
       \f[
       \mbox{log1p}(x) =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \ln(1+x)& \mbox{if } x\geq -1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{log1p}(x)}{\partial x} =
       \begin{cases}
         \textrm{NaN} & \mbox{if } x < -1\\
         \frac{1}{1+x} & \mbox{if } x\geq -1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
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
