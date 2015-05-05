#ifndef STAN_MATH_PRIM_SCAL_FUN_CORR_CONSTRAIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_CORR_CONSTRAIN_HPP

#include <stan/math/prim/scal/fun/log1m.hpp>
#include <cmath>

namespace stan {

  namespace math {

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is the hyperbolic tangent function,
     *
     * <p>\f$f(x) = \tanh x = \frac{\exp(2x) - 1}{\exp(2x) + 1}\f$.
     *
     * @param x Scalar input.
     * @return Result of transforming the input to fall between -1 and 1.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x) {
      return tanh(x);
    }

    /**
     * Return the result of transforming the specified scalar to have
     * a valid correlation value between -1 and 1 (inclusive).
     *
     * <p>The transform used is as specified for
     * <code>corr_constrain(T)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$\log | \frac{d}{dx} \tanh x  | = \log (1 - \tanh^2 x)\f$.
     *
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_constrain(const T x, T& lp) {
      using stan::math::log1m;
      T tanh_x = tanh(x);
      lp += log1m(tanh_x * tanh_x);
      return tanh_x;
    }

  }

}

#endif
