#ifndef STAN__AGRAD__FWD__FUNCTIONS__ATAN2_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ATAN2_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    atan2(const fvar<T>& x1, const fvar<T>& x2) {
      using ::atan2;
      using stan::math::square;
      return fvar<T>(atan2(x1.val_, x2.val_), 
                     (x1.d_ * x2.val_ - x1.val_ * x2.d_) / 
                     (square(x2.val_) + square(x1.val_)));
    }

    template <typename T>
    inline
    fvar<T>
    atan2(const double x1, const fvar<T>& x2) {
      using ::atan2;
      using stan::math::square;
      return fvar<T>(atan2(x1, x2.val_), 
                     (-x1 * x2.d_) / (square(x1) + square(x2.val_)));
    }

    template <typename T>
    inline
    fvar<T>
    atan2(const fvar<T>& x1, const double x2) {
      using ::atan2;
      using stan::math::square;
      return fvar<T>(atan2(x1.val_, x2), 
                     (x1.d_ * x2) / (square(x2) + square(x1.val_)));
    }
  }
}
#endif
