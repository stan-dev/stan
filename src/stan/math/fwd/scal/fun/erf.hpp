#ifndef STAN__MATH__FWD__SCAL__FUN__ERF_HPP
#define STAN__MATH__FWD__SCAL__FUN__ERF_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <math.h>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    erf(const fvar<T>& x) {
      using ::sqrt;
      using ::exp;
      using ::erf;
      using stan::math::square;
      return fvar<T>(erf(x.val_), x.d_ * exp(-square(x.val_)) 
                                  * stan::math::TWO_OVER_SQRT_PI);
    }
  }
}
#endif
