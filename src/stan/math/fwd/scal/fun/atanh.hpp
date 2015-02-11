#ifndef STAN__AGRAD__FWD__FUNCTIONS__ATANH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ATANH_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    atanh(const fvar<T>& x) {
      using ::atanh;
      using stan::math::square;
      return fvar<T>(atanh(x.val_), x.d_ / (1 - square(x.val_)));
    }
  }
}
#endif
