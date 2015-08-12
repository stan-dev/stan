#ifndef STAN_MATH_PRIM_SCAL_FUN_LOG_INV_LOGIT_HPP
#define STAN_MATH_PRIM_SCAL_FUN_LOG_INV_LOGIT_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the natural logarithm of the inverse logit of the
     * specified argument.
     *
     *
       \f[
       \mbox{log\_inv\_logit}(x) =
       \begin{cases}
         \ln\left(\frac{1}{1+\exp(-x)}\right)& \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\, \mbox{log\_inv\_logit}(x)}{\partial x} =
       \begin{cases}
         \frac{1}{1+\exp(x)} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @tparam T Scalar type
     * @param u Input.
     * @return log of the inverse logit of the input.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log_inv_logit(const T& u) {
      using std::exp;
      if (u < 0.0)
        return u - log1p(exp(u));  // prevent underflow
      return -log1p(exp(-u));
    }

  }
}

#endif
