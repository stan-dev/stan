#ifndef STAN_MATH_FWD_SCAL_FUN_HYPOT_HPP
#define STAN_MATH_FWD_SCAL_FUN_HYPOT_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> hypot(const fvar<T>& x1, const fvar<T>& x2) {
      using ::hypot;
      using std::sqrt;
      using stan::math::inv;
      T u = hypot(x1.val_, x2.val_);
      return fvar<T>(u, (x1.d_ * x1.val_ + x2.d_ * x2.val_) * inv(u));
    }

    template <typename T>
    inline fvar<T> hypot(const fvar<T>& x1, const double x2) {
      using ::hypot;
      using std::sqrt;
      using stan::math::inv;
      T u = hypot(x1.val_, x2);
      return fvar<T>(u, (x1.d_ * x1.val_) * inv(u));
    }

    template <typename T>
    inline fvar<T> hypot(const double x1, const fvar<T>& x2) {
      using ::hypot;
      using std::sqrt;
      using stan::math::inv;
      T u = hypot(x1, x2.val_);
      return fvar<T>(u, (x2.d_ * x2.val_) * inv(u));
    }

  }
}
#endif
