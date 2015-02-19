#ifndef STAN__MATH__FWD__SCAL__FUN__TRUNC_HPP
#define STAN__MATH__FWD__SCAL__FUN__TRUNC_HPP

#include <math.h>
#include <stan/math/fwd/core/fvar.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    trunc(const fvar<T>& x) {
      using ::trunc;
      return fvar<T>(trunc(x.val_), 0);
    }

  }
}
#endif
