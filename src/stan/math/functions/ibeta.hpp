#ifndef STAN__MATH__FUNCTIONS__IBETA_HPP
#define STAN__MATH__FUNCTIONS__IBETA_HPP

#include <boost/math/special_functions/beta.hpp>

namespace stan {
  namespace math {

    /** 
     * The normalized incomplete beta function of a, b, and x.
     *
     * Used to compute the cumulative density function for the beta
     * distribution.
     * 
     * @param a Shape parameter.
     * @param b Shape parameter.
     * @param x Random variate.
     * 
     * @return The normalized incomplete beta function.
     */
    inline double ibeta(const double a,
                        const double b,
                        const double x) {
      return boost::math::ibeta(a, b, x);
    }

  }
}

#endif
