#ifndef STAN__MATH__FUNCTIONS__INV_LOGIT_HPP
#define STAN__MATH__FUNCTIONS__INV_LOGIT_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {

  namespace math {

    /**
     * Returns the inverse logit function applied to the argument.
     *
     * The inverse logit function is defined by
     *
     * \f$\mbox{logit}^{-1}(x) = \frac{1}{1 + \exp(-x)}\f$.
     *
     * This function can be used to implement the inverse link function
     * for logistic regression.
     *
     * The inverse to this function is <code>stan::math::logit</code>.
     * 
     * @param a Argument.
     * @return Inverse logit of argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    inv_logit(const T a) {
      using std::exp;
      return 1.0 / (1.0 + exp(-a));
    }

  }
}

#endif
