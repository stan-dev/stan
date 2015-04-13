#ifndef STAN_MATH_FWD_SCAL_FUN_CEIL_HPP
#define STAN_MATH_FWD_SCAL_FUN_CEIL_HPP

#include <stan/math/fwd/core.hpp>

#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    ceil(const fvar<T>& x) {
      using ::ceil;
        return fvar<T>(ceil(x.val_), 0);
    }
  }
}
#endif
