#ifndef STAN_MATH_FWD_SCAL_FUN_LOG_FALLING_FACTORIAL_HPP
#define STAN_MATH_FWD_SCAL_FUN_LOG_FALLING_FACTORIAL_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/log_falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    template<typename T>
    inline fvar<T>
    log_falling_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<T>(log_falling_factorial(x.val_, n.val_),
                     digamma(x.val_ + 1) * x.d_ - digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<T>
    log_falling_factorial(const double x, const fvar<T>& n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<T>(log_falling_factorial(x, n.val_),
                     -digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<T>
    log_falling_factorial(const fvar<T>& x, const double n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<T>(log_falling_factorial(x.val_, n),
                     digamma(x.val_ + 1) * x.d_);
    }
  }
}
#endif
