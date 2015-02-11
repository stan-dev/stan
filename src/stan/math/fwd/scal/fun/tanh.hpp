#ifndef STAN__AGRAD__FWD__FUNCTIONS__TANH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__TANH_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    tanh(const fvar<T>& x) {
      using std::tanh;
      T u = tanh(x.val_);
      return fvar<T>(u, x.d_ * (1 - u * u));
    }
  }
}
#endif
