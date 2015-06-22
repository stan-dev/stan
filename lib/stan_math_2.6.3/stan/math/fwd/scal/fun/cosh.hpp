#ifndef STAN_MATH_FWD_SCAL_FUN_COSH_HPP
#define STAN_MATH_FWD_SCAL_FUN_COSH_HPP

#include <stan/math/fwd/core.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline
    fvar<T>
    cosh(const fvar<T>& x) {
      using std::sinh;
      using std::cosh;
      return fvar<T>(cosh(x.val_), x.d_ * sinh(x.val_));
    }

  }
}
#endif
