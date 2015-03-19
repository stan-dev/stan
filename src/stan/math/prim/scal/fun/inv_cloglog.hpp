#ifndef STAN__MATH__PRIM__SCAL__FUN__INV_CLOGLOG_HPP
#define STAN__MATH__PRIM__SCAL__FUN__INV_CLOGLOG_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * The inverse complementary log-log function.
     *
     * The function is defined by
     *
     * <code>inv_cloglog(x) = 1 - exp(-exp(x))</code>.
     *
     * This function can be used to implement the inverse link
     * function for complementary-log-log regression.
     *
     *
       \f[
       \mbox{inv\_cloglog}(y) =
       \begin{cases}
         \mbox{cloglog}^{-1}(y) & \mbox{if } -\infty\leq y \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{inv\_cloglog}(y)}{\partial y} =
       \begin{cases}
         \frac{\partial\, \mbox{cloglog}^{-1}(y)}{\partial y} & \mbox{if } -\infty\leq y\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } y = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \mbox{cloglog}^{-1}(y) = 1 - \exp \left( - \exp(y) \right)
       \f]

       \f[
       \frac{\partial \, \mbox{cloglog}^{-1}(y)}{\partial y} = \exp(y-\exp(y))
       \f]
     *
     * @param x Argument.
     * @return Inverse complementary log-log of the argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    inv_cloglog(T x) {
      using std::exp;
      return 1 - exp(-exp(x));
    }

  }
}

#endif
