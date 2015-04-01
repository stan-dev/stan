#ifndef STAN__MATH__FWD__SCAL__FUN__ERFC_HPP
#define STAN__MATH__FWD__SCAL__FUN__ERFC_HPP

#include <math.h>
#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    erfc(const fvar<T>& x) {
      using ::erfc;
      using ::sqrt;
      using ::exp;
      using stan::math::square;
      return fvar<T>(erfc(x.val_), -x.d_ * exp(-square(x.val_))
                                    * stan::math::TWO_OVER_SQRT_PI);
    }
  }
}
#endif
