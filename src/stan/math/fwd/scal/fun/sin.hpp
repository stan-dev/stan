#ifndef STAN__MATH__FWD__SCAL__FUN__SIN_HPP
#define STAN__MATH__FWD__SCAL__FUN__SIN_HPP

#include <stan/math/fwd/core.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    sin(const fvar<T>& x) {
      using std::sin;
      using std::cos;
      return fvar<T>(sin(x.val_),
                     x.d_ * cos(x.val_));
    }
  }
}
#endif
