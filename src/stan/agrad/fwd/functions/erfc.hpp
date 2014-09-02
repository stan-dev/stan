#ifndef STAN__AGRAD__FWD__FUNCTIONS__ERFC_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ERFC_HPP

#include <math.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>

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
