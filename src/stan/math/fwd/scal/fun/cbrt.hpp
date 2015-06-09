#ifndef STAN_MATH_FWD_SCAL_FUN_CBRT_HPP
#define STAN_MATH_FWD_SCAL_FUN_CBRT_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {
  namespace math {

    template <typename T>
    inline
    fvar<T>
    cbrt(const fvar<T>& x) {
      using ::cbrt;
      using stan::math::square;
      return fvar<T>(cbrt(x.val_),
                     x.d_ / (square(cbrt(x.val_)) * 3.0));
    }

  }
}
#endif
