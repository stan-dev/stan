#ifndef __STAN__AGRAD__FWD__SQUARE__HPP__
#define __STAN__AGRAD__FWD__SQUARE__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>

namespace stan{

  namespace agrad{

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
