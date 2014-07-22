#ifndef STAN__AGRAD__FWD__FUNCTIONS__SQUARE_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__SQUARE_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace agrad {

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
