#ifndef STAN__MATH__FUNCTIONS__PHI_HPP
#define STAN__MATH__FUNCTIONS__PHI_HPP

#include <boost/math/tools/promotion.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <stan/math/constants.hpp>
#include <stan/error_handling/scalar/check_not_nan.hpp>

namespace stan {
  namespace math {

    /**
     * The unit normal cumulative distribution function.  
     *
     * The return value for a specified input is the probability that
     * a random unit normal variate is less than or equal to the
     * specified value, defined by
     *
     * \f$\Phi(x) = \int_{-\infty}^x \mbox{\sf Norm}(x|0,1) \ dx\f$
     *
     * This function can be used to implement the inverse link function
     * for probit regression.  
     *
     * Phi will underflow to 0 below -37.5 and overflow to 1 above 8
     *
     * @param x Argument.
     * @return Probability random sample is less than or equal to argument. 
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    Phi(const T x) {
      // overridden in fvar and var, so can hard-code boost versions
      // here for scalars only
      using stan::error_handling::check_not_nan;
      
      check_not_nan("Phi",  "x", x);
      if (x < -37.5)
        return 0;
      else if (x < -5.0)
        return 0.5 * boost::math::erfc(-INV_SQRT_2 * x);
      else if (x > 8.25)
        return 1;
      else
        return 0.5 * (1.0 + boost::math::erf(INV_SQRT_2 * x));
    }

  }
}

#endif
