#ifndef STAN__AGRAD__FWD__FUNCTIONS__TRUNC_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__TRUNC_HPP

#include <math.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
