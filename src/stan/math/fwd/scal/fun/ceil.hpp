#ifndef STAN__MATH__FWD__SCAL__FUN__CEIL_HPP
#define STAN__MATH__FWD__SCAL__FUN__CEIL_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    ceil(const fvar<T>& x) {
      using ::ceil;
        return fvar<T>(ceil(x.val_), 0);
    }
  }
}
#endif
