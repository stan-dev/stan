#ifndef __STAN__DIFF__FWD__TANH__HPP__
#define __STAN__DIFF__FWD__TANH__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    tanh(const fvar<T>& x) {
      using std::tanh;
      return fvar<T>(tanh(x.val_), x.d_ * (1 - tanh(x.val_) * tanh(x.val_)));
    }
  }
}
#endif
