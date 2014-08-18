#ifndef STAN__MATH__FUNCTIONS__IBETA_HPP
#define STAN__MATH__FUNCTIONS__IBETA_HPP

#include <boost/math/special_functions/beta.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>

namespace stan {
  namespace math {

    /** 
     * The normalized incomplete beta function of a, b, and x.
     *
     * Used to compute the cumulative density function for the beta
     * distribution.
     * 
     * @param a Shape parameter a <= 0; a and b can't both be 0
     * @param b Shape parameter b <= 0
     * @param x Random variate. 0 <= x <= 1
     * @throws if constraints are violated
     * 
     * @return The normalized incomplete beta function.
     */
    inline double ibeta(const double a,
                        const double b,
                        const double x) {
      check_not_nan("ibeta(%1%)", a, "a", static_cast<double*>(0));
      check_not_nan("ibeta(%1%)", b, "b", static_cast<double*>(0));
      check_not_nan("ibeta(%1%)", x, "x", static_cast<double*>(0));
      return boost::math::ibeta(a, b, x);
    }

  }
}

#endif
