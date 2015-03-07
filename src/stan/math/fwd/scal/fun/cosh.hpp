#ifndef STAN__MATH__FWD__SCAL__FUN__COSH_HPP
#define STAN__MATH__FWD__SCAL__FUN__COSH_HPP

#include <stan/math/fwd/core.hpp>

#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    cosh(const fvar<T>& x) {
      using ::sinh;
      using ::cosh;
      return fvar<T>(cosh(x.val_), x.d_ * sinh(x.val_));
    }
  }
}
#endif
