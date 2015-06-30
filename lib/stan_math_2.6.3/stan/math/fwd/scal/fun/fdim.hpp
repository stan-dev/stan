#ifndef STAN_MATH_FWD_SCAL_FUN_FDIM_HPP
#define STAN_MATH_FWD_SCAL_FUN_FDIM_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/fun/fdim.hpp>

namespace stan {
  namespace math {

    template <typename T>
    inline fvar<T> fdim(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::fdim;
      using std::floor;
      if (x1.val_ < x2.val_)
        return fvar<T>(fdim(x1.val_, x2.val_), 0);
      else
        return fvar<T>(fdim(x1.val_, x2.val_),
                       x1.d_ - x2.d_ * floor(x1.val_ / x2.val_));
    }

    template <typename T>
    inline fvar<T> fdim(const fvar<T>& x1, const double x2) {
      using stan::math::fdim;
      using std::floor;
      if (x1.val_ < x2)
        return fvar<T>(fdim(x1.val_, x2), 0);
      else
        return fvar<T>(fdim(x1.val_, x2), x1.d_);
    }

    template <typename T>
    inline fvar<T> fdim(const double x1, const fvar<T>& x2) {
      using stan::math::fdim;
      using std::floor;
      if (x1 < x2.val_)
        return fvar<T>(fdim(x1, x2.val_), 0);
      else
        return fvar<T>(fdim(x1, x2.val_), x2.d_ * -floor(x1 / x2.val_));
    }

  }
}
#endif
