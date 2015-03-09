#ifndef STAN__MATH__FWD__SCAL__FUN__ATANH_HPP
#define STAN__MATH__FWD__SCAL__FUN__ATANH_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    atanh(const fvar<T>& x) {
      using ::atanh;
      using stan::math::square;
      return fvar<T>(atanh(x.val_), x.d_ / (1 - square(x.val_)));
    }
  }
}
#endif
