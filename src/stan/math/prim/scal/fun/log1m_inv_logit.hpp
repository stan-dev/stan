#ifndef STAN__MATH__PRIM__SCAL__FUN__LOG1M_INV_LOGIT_HPP
#define STAN__MATH__PRIM__SCAL__FUN__LOG1M_INV_LOGIT_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the natural logarithm of 1 minus the inverse logit
     * of the specified argument.
     *
     *
       \f[
       \mbox{log1m\_inv\_logit}(x) =
       \begin{cases}
         -\ln(\exp(x)+1) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{log1m\_inv\_logit}(x)}{\partial x} =
       \begin{cases}
         -\frac{\exp(x)}{\exp(x)+1} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @tparam T Scalar type
     * @param u Input.
     * @return log of 1 minus the inverse logit of the input.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1m_inv_logit(const T u) {
      using std::exp;
      if (u > 0.0)
        return -u - log1p(exp(-u));  // prevent underflow
      return -log1p(exp(u));
    }

  }
}

#endif
