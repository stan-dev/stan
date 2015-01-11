#ifndef STAN__AGRAD__FWD__FUNCTIONS__TAN_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__TAN_HPP

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
