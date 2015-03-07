#ifndef STAN__MATH__FWD__SCAL__FUN__SINH_HPP
#define STAN__MATH__FWD__SCAL__FUN__SINH_HPP

#include <stan/math/fwd/core.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    sinh(const fvar<T>& x) {
      using std::sinh;
      using std::cosh;
      return fvar<T>(sinh(x.val_),
                     x.d_ * cosh(x.val_));
    }
  }
}
#endif
