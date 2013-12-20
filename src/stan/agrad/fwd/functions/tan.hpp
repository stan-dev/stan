#ifndef __STAN__AGRAD__FWD__FUNCTIONS__TAN_HPP__
#define __STAN__AGRAD__FWD__FUNCTIONS__TAN_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
