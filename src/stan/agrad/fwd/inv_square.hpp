#ifndef __STAN__AGRAD__FWD__INV_SQUARE_HPP__
#define __STAN__AGRAD__FWD__INV_SQUARE_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    inv_square(const fvar<T>& x) {
      return fvar<T>(1 / (x.val_ * x.val_), -2 * x.d_ / (x.val_ * x.val_ * x.val_));
    }
  }
}
#endif
