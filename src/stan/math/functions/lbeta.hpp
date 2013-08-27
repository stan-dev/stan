#ifndef __STAN__MATH__FUNCTIONS__LBETA_HPP__
#define __STAN__MATH__FUNCTIONS__LBETA_HPP__

#include <boost/math/tools/promotion.hpp>

#include <boost/math/special_functions/gamma.hpp>

namespace stan {
  namespace math {

    /**
     * Return the log of the beta function applied to the specified
     * arguments.
     *
     * The beta function is defined for \f$a > 0\f$ and \f$b > 0\f$ by
     *
     * \f$\mbox{B}(a,b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a+b)}\f$.
     *
     * This function returns its log,
     *
     * \f$\log \mbox{B}(a,b) = \log \Gamma(a) + \log \Gamma(b) - \log \Gamma(a+b)\f$.
     *
     * See boost::math::lgamma() for the double-based and stan::diff for the
     * variable-based log Gamma function.
     * 
     * @param a First value
     * @param b Second value
     * @return Log of the beta function applied to the two values.
     * @tparam T1 Type of first value.
     * @tparam T2 Type of second value.
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    lbeta(const T1 a, const T2 b) {
      using boost::math::lgamma;
      return lgamma(a)
        + lgamma(b)
        - lgamma(a + b);
    }

  }
}

#endif
