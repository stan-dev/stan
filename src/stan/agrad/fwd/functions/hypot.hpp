#ifndef STAN__AGRAD__FWD__FUNCTIONS__HYPOT_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__HYPOT_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    hypot(const fvar<T>& x1, const fvar<T>& x2) {
      using ::hypot;
      using ::sqrt;
      using stan::math::inv;
      T u = hypot(x1.val_, x2.val_);
      return fvar<T>(u, (x1.d_ * x1.val_ + x2.d_ * x2.val_) * inv(u));
    }

    template <typename T>
    inline
    fvar<T>
    hypot(const fvar<T>& x1, const double x2) {
      using ::hypot;
      using ::sqrt;
      using stan::math::inv;
      T u = hypot(x1.val_, x2);
      return fvar<T>(u, (x1.d_ * x1.val_) * inv(u));
    }

    template <typename T>
    inline
    fvar<T>
    hypot(const double x1, const fvar<T>& x2) {
      using ::hypot;
      using ::sqrt;
      using stan::math::inv;
      T u = hypot(x1, x2.val_);
      return fvar<T>(u, (x2.d_ * x2.val_) * inv(u));
    }

  }
}
#endif
