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
