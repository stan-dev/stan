#ifndef STAN_MATH_PRIM_SCAL_FUN_POSITIVE_CONSTRAIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_POSITIVE_CONSTRAIN_HPP

#include <cmath>

namespace stan {

  namespace math {

    /**
     * Return the positive value for the specified unconstrained input.
     *
     * <p>The transform applied is
     *
     * <p>\f$f(x) = \exp(x)\f$.
     *
     * @param x Arbitrary input scalar.
     * @return Input transformed to be positive.
     */
    template <typename T>
    inline
    T positive_constrain(const T x) {
      return exp(x);
    }

    /**
     * Return the positive value for the specified unconstrained input,
     * incrementing the scalar reference with the log absolute
     * Jacobian determinant.
     *
     * <p>See <code>positive_constrain(T)</code> for details
     * of the transform.  The log absolute Jacobian determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \mbox{exp}(x) |
     *    = \log | \mbox{exp}(x) | =  x\f$.
     *
     * @param x Arbitrary input scalar.
     * @param lp Log probability reference.
     * @return Input transformed to be positive.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T positive_constrain(const T x, T& lp) {
      lp += x;
      return exp(x);
    }


  }

}

#endif
