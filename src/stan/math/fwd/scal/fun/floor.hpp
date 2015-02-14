#ifndef STAN__MATH__FWD__SCAL__FUN__FLOOR_HPP
#define STAN__MATH__FWD__SCAL__FUN__FLOOR_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    floor(const fvar<T>& x) {
      using ::floor;
        return fvar<T>(floor(x.val_), 0);
    }
  }
}
#endif
