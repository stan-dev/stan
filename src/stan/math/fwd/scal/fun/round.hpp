#ifndef STAN_MATH_FWD_SCAL_FUN_ROUND_HPP
#define STAN_MATH_FWD_SCAL_FUN_ROUND_HPP

#include <stan/math/fwd/core.hpp>
#include <cmath>


namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> round(const fvar<T>& x) {
      using std::round;
      return fvar<T>(round(x.val_), 0);
    }

  }
}
#endif
