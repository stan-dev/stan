#ifndef STAN_MATH_FWD_SCAL_FUN_FALLING_FACTORIAL_HPP
#define STAN_MATH_FWD_SCAL_FUN_FALLING_FACTORIAL_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    template<typename T>
    inline fvar<T>
    falling_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      T falling_fact(falling_factorial(x.val_, n.val_));
      return fvar<T>(falling_fact, falling_fact * digamma(x.val_ + 1) * x.d_
                     - falling_fact * digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<T>
    falling_factorial(const fvar<T>& x, const double n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      T falling_fact(falling_factorial(x.val_, n));
      return fvar<T>(falling_fact,
                     falling_fact * digamma(x.val_ + 1) * x.d_);
    }

    template<typename T>
    inline fvar<T>
    falling_factorial(const double x, const fvar<T>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      T falling_fact(falling_factorial(x, n.val_));
      return fvar<T>(falling_fact,
                     -falling_fact * digamma(n.val_ + 1) * n.d_);
    }
  }
}
#endif
