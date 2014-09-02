#ifndef STAN__AGRAD__FWD__FUNCTIONS__CEIL_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__CEIL_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
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
