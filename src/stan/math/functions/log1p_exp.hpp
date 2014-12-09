#ifndef STAN__MATH__FUNCTIONS__LOG1P_EXP_HPP
#define STAN__MATH__FUNCTIONS__LOG1P_EXP_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/functions/log1p.hpp>

namespace stan {
  namespace math {

    /**
     * Calculates the log of 1 plus the exponential of the specified
     * value without overflow.                  
     *
     * This function is related to other special functions by:
     *
     * <code>log1p_exp(x) </code>
     *
     * <code> = log1p(exp(a))</code>
     *
     * <code> = log(1 + exp(x))</code>
     *
     * <code> = log_sum_exp(0,x)</code>.
     *
     *
       \f[
       \mbox{log1p\_exp}(x) = 
       \begin{cases}
         \ln(1+\exp(x)) & \mbox{if } -\infty\leq x \leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
       
       \f[
       \frac{\partial\,\mbox{log1p\_exp}(x)}{\partial x} = 
       \begin{cases}
         \frac{\exp(x)}{1+\exp(x)} & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1p_exp(const T a) {
      using std::exp;
      // like log_sum_exp below with b=0.0; prevents underflow
      if (a > 0.0)
        return a + log1p(exp(-a)); 
      return log1p(exp(a));
    }

  }
}

#endif
