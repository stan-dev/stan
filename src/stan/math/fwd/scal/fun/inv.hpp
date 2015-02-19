#ifndef STAN__MATH__FWD__SCAL__FUN__INV_HPP
#define STAN__MATH__FWD__SCAL__FUN__INV_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {

  namespace agrad {

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
