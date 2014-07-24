#ifndef STAN__AGRAD__FWD__FUNCTIONS__INV_SQUARE_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__INV_SQUARE_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>

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
