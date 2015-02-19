#ifndef STAN__MATH__FWD__SCAL__FUN__SQUARE_HPP
#define STAN__MATH__FWD__SCAL__FUN__SQUARE_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/square.hpp>

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
