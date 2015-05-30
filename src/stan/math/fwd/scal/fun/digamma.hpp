#ifndef STAN_MATH_FWD_SCAL_FUN_DIGAMMA_HPP
#define STAN_MATH_FWD_SCAL_FUN_DIGAMMA_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/trigamma.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    digamma(const fvar<T>& x) {
      using stan::math::digamma;
      using stan::math::trigamma;
      return fvar<T>(digamma(x.val_), x.d_ * trigamma(x.val_));
    }
  }
}
#endif
