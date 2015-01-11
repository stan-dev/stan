#ifndef STAN__AGRAD__FWD__FUNCTIONS__ACOS_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ACOS_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    acos(const fvar<T>& x) {
      using ::acos;
      using ::sqrt;
      using stan::math::square;

      return fvar<T>(acos(x.val_), x.d_ / -sqrt(1 - square(x.val_)));
    }
  }
}
#endif
