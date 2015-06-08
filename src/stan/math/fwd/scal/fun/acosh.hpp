#ifndef STAN_MATH_FWD_SCAL_FUN_ACOSH_HPP
#define STAN_MATH_FWD_SCAL_FUN_ACOSH_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> acosh(const fvar<T>& x) {
      using ::acosh;
      using stan::math::square;
      using std::sqrt;
      using stan::math::NOT_A_NUMBER;
      return fvar<T>(acosh(x.val_),
                     x.d_ / sqrt(square(x.val_) - 1));
    }

  }
}
#endif
