#ifndef STAN__MATH__FWD__SCAL__FUN__ASINH_HPP
#define STAN__MATH__FWD__SCAL__FUN__ASINH_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    asinh(const fvar<T>& x) {
      using ::asinh;
      using std::sqrt;
      using stan::math::square;
      return fvar<T>(asinh(x.val_), x.d_ / sqrt(square(x.val_) + 1));
    }
  }
}
#endif
