#ifndef STAN_MATH_PRIM_SCAL_FUN_PROB_FREE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_PROB_FREE_HPP

#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/fun/logit.hpp>

namespace stan {

  namespace math {

    /**
     * Return the free scalar that when transformed to a probability
     * produces the specified scalar.
     *
     * <p>The function that reverses the constraining transform
     * specified in <code>prob_constrain(T)</code> is the logit
     * function,
     *
     * <p>\f$f^{-1}(y) = \mbox{logit}(y) = \frac{1 - y}{y}\f$.
     *
     * @param y Scalar input.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is less than 0 or greater than 1.
     */
    template <typename T>
    inline
    T prob_free(const T y) {
      using stan::math::logit;
      stan::math::check_bounded<T, double, double>
        ("stan::math::prob_free", "Probability variable",
         y, 0, 1);
      return logit(y);
    }

  }

}

#endif
