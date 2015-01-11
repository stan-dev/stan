#ifndef STAN__AGRAD__FWD__FUNCTIONS__EXPM1_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__EXPM1_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    expm1(const fvar<T>& x) {
      using ::expm1;
      using ::exp;
      return fvar<T>(expm1(x.val_), x.d_ * exp(x.val_));
    } 
  }
}
#endif
