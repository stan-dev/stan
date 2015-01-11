#ifndef STAN__AGRAD__FWD__FUNCTIONS__SIN_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__SIN_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
