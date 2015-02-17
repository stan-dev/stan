#ifndef STAN__MATH__FWD__SCAL__FUN__ATAN_HPP
#define STAN__MATH__FWD__SCAL__FUN__ATAN_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    atan(const fvar<T>& x) {
      using ::atan;
      using stan::math::square;
      return fvar<T>(atan(x.val_), x.d_ / (1 + square(x.val_)));
    }
  }
}
#endif
