#ifndef STAN_MATH_FWD_SCAL_FUN_EXP_HPP
#define STAN_MATH_FWD_SCAL_FUN_EXP_HPP

#include <stan/math/fwd/core.hpp>

#include <math.h>

namespace stan {

  namespace math {


    template <typename T>
    inline
    fvar<T>
    exp(const fvar<T>& x) {
      using ::exp;
      return fvar<T>(exp(x.val_), x.d_ * exp(x.val_));
    }
  }
}
#endif
