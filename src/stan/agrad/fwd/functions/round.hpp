#ifndef STAN__AGRAD__FWD__FUNCTIONS__ROUND_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ROUND_HPP

#include <math.h>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    round(const fvar<T>& x) {
      using ::round;
        return fvar<T>(round(x.val_), 0);
    }

  }
}
#endif
