#ifndef STAN__MATH__FUNCTIONS__INV_CLOGLOG_HPP
#define STAN__MATH__FUNCTIONS__INV_CLOGLOG_HPP

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
