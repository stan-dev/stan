#ifndef STAN__MATH__FWD__SCAL__FUN__ASIN_HPP
#define STAN__MATH__FWD__SCAL__FUN__ASIN_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    asin(const fvar<T>& x) {
      using ::asin;
      using std::sqrt;
      using stan::math::square;
      return fvar<T>(asin(x.val_), x.d_ / sqrt(1 - square(x.val_)));
    }
  }
}
#endif
