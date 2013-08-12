#ifndef __STAN__DIFF__FWD__SQUARE__HPP__
#define __STAN__DIFF__FWD__SQUARE__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    square(const fvar<T>& x) {
      using stan::math::square;
      return fvar<T>(square(x.val_),
                     x.d_ * 2 * x.val_);
    }
  }
}
#endif
