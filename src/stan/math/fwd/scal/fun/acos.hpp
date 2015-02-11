#ifndef STAN__AGRAD__FWD__FUNCTIONS__ACOS_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ACOS_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
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
