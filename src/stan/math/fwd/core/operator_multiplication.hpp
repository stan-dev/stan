#ifndef STAN__MATH__FWD__CORE__OPERATOR_MULTIPLICATION_HPP
#define STAN__MATH__FWD__CORE__OPERATOR_MULTIPLICATION_HPP

#include <stan/math/fwd/core/fvar.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    operator*(const fvar<T>& x1, const fvar<T>& x2) {
      return fvar<T>(x1.val_ * x2.val_,
                     x1.d_ * x2.val_ + x1.val_ * x2.d_);
    }

    template <typename T>
    inline
    fvar<T>
    operator*(const double x1, const fvar<T>& x2) {
      return fvar<T>(x1 * x2.val_, x1 * x2.d_);
    }

    template <typename T>
    inline
    fvar<T>
    operator*(const fvar<T>& x1, const double x2) {
      return fvar<T>(x1.val_ * x2, x1.d_ * x2);
    }
  }
}
#endif
