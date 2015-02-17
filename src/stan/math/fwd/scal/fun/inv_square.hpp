#ifndef STAN__MATH__FWD__SCAL__FUN__INV_SQUARE_HPP
#define STAN__MATH__FWD__SCAL__FUN__INV_SQUARE_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    inv_square(const fvar<T>& x) {
      using stan::math::square;
      T square_x(square(x.val_));
      return fvar<T>(1 / square_x, -2 * x.d_ / (square_x * x.val_));
    }
  }
}
#endif
