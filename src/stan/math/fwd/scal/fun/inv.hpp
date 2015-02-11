#ifndef STAN__AGRAD__FWD__FUNCTIONS__INV_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__INV_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
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
