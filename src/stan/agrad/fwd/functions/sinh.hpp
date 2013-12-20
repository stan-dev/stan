#ifndef __STAN__AGRAD__FWD__FUNCTIONS__SINH_HPP__
#define __STAN__AGRAD__FWD__FUNCTIONS__SINH_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
