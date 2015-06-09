#ifndef STAN_MATH_FWD_SCAL_FUN_TRUNC_HPP
#define STAN_MATH_FWD_SCAL_FUN_TRUNC_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>


namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> trunc(const fvar<T>& x) {
      using ::trunc;
      return fvar<T>(trunc(x.val_), 0);
    }

  }
}
#endif
