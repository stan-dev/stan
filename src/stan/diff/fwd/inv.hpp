#ifndef __STAN__DIFF__FWD__INV__HPP__
#define __STAN__DIFF__FWD__INV__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace diff {

    template <typename T>
    inline
    fvar<T>
    inv(const fvar<T>& x) {
      using stan::math::square;
      return fvar<T>(1 / x.val_, -x.d_ / square(x.val_));
    }
  }
}
#endif
