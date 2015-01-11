#ifndef STAN__AGRAD__FWD__FUNCTIONS__ATAN_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ATAN_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>
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
