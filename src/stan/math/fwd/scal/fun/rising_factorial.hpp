#ifndef STAN_MATH_FWD_SCAL_FUN_RISING_FACTORIAL_HPP
#define STAN_MATH_FWD_SCAL_FUN_RISING_FACTORIAL_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/rising_factorial.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <iostream>

namespace stan {

  namespace math {

    template<typename T>
    inline
    fvar<T>
    rising_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::rising_factorial;

      T rising_fact(rising_factorial(x.val_, n.val_));
      return fvar<T>(rising_fact,
                     rising_fact * (digamma(x.val_ + n.val_)
                                    * (x.d_ + n.d_) - digamma(x.val_) * x.d_));
    }

    template<typename T>
    inline
    fvar<T>
    rising_factorial(const fvar<T>& x, const double n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      T rising_fact(rising_factorial(x.val_, n));
      return fvar<T>(rising_fact,
                     rising_fact * x.d_
                     * (digamma(x.val_ + n) - digamma(x.val_)));
    }

    template<typename T>
    inline
    fvar<T>
    rising_factorial(const double x, const fvar<T>& n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      T rising_fact(rising_factorial(x, n.val_));
      return fvar<T>(rising_fact,
                     rising_fact * (digamma(x + n.val_) * n.d_));
    }
  }
}
#endif
