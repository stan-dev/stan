#ifndef STAN_MATH_FWD_SCAL_FUN_EXPM1_HPP
#define STAN_MATH_FWD_SCAL_FUN_EXPM1_HPP

#include <stan/math/fwd/core.hpp>

#include <math.h>

namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    expm1(const fvar<T>& x) {
      using ::expm1;
      using ::exp;
      return fvar<T>(expm1(x.val_), x.d_ * exp(x.val_));
    }
  }
}
#endif
