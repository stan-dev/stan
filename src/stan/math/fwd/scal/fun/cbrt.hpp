#ifndef STAN__MATH__FWD__SCAL__FUN__CBRT_HPP
#define STAN__MATH__FWD__SCAL__FUN__CBRT_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    cbrt(const fvar<T>& x) {
      using ::cbrt;
      using stan::math::square;
      return fvar<T>(cbrt(x.val_),
                     x.d_ / (square(cbrt(x.val_)) * 3.0));
    }
  }
}
#endif
