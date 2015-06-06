#ifndef STAN_MATH_FWD_SCAL_FUN_ASINH_HPP
#define STAN_MATH_FWD_SCAL_FUN_ASINH_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> asinh(const fvar<T>& x) {
      using ::asinh;
      using std::sqrt;
      using stan::math::square;
      return fvar<T>(asinh(x.val_), x.d_ / sqrt(square(x.val_) + 1));
    }

  }
}
#endif
