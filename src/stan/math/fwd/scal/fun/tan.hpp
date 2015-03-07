#ifndef STAN__MATH__FWD__SCAL__FUN__TAN_HPP
#define STAN__MATH__FWD__SCAL__FUN__TAN_HPP

#include <stan/math/fwd/core.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    tan(const fvar<T>& x) {
      using std::cos;
      using std::tan;
      return fvar<T>(tan(x.val_), x.d_ / (cos(x.val_) * cos(x.val_)));
    }
  }
}
#endif
