#ifndef STAN__MATH__FUNCTIONS__LOGIT_HPP
#define STAN__MATH__FUNCTIONS__LOGIT_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the logit function applied to the
     * argument. 
     *
     * The logit function is defined as for \f$x \in [0,1]\f$ by
     * returning the log odds of \f$x\f$ treated as a probability,
     *
     * \f$\mbox{logit}(x) = \log \left( \frac{x}{1 - x} \right)\f$.
     *
     * The inverse to this function is <code>stan::math::inv_logit</code>.
     *
     *
       \f[
       \mbox{logit}(x) = 
       \begin{cases}
         \textrm{NaN}& \mbox{if } x < 0 \textrm{ or } x > 1\\
         \ln\frac{x}{1-x} & \mbox{if } 0\leq x \leq 1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
   
       \f[
       \frac{\partial\,\mbox{logit}(x)}{\partial x} = 
       \begin{cases}
         \textrm{NaN}& \mbox{if } x < 0 \textrm{ or } x > 1\\
         \frac{1}{x-x^2}& \mbox{if } 0\leq x\leq 1 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param a Argument.
     * @return Logit of the argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    logit(const T a) {
      using std::log;
      return log(a / (1.0 - a));
    }

  }
}

#endif
