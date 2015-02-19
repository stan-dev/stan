#ifndef STAN__MATH__FWD__SCAL__FUN__TANH_HPP
#define STAN__MATH__FWD__SCAL__FUN__TANH_HPP

#include <stan/math/fwd/core/fvar.hpp>


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
