#ifndef STAN_MATH_FWD_SCAL_FUN_ATANH_HPP
#define STAN_MATH_FWD_SCAL_FUN_ATANH_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> atanh(const fvar<T>& x) {
      using ::atanh;
      using stan::math::square;
      return fvar<T>(atanh(x.val_), x.d_ / (1 - square(x.val_)));
    }

  }
}
#endif
