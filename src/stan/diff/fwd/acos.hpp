#ifndef __STAN__AGRAD__FWD__ACOS__HPP__
#define __STAN__AGRAD__FWD__ACOS__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    acos(const fvar<T>& x) {
      using std::acos;
      using std::sqrt;
      return fvar<T>(acos(x.val_), x.d_ / -sqrt(1 - x.val_ * x.val_));
    }
  }
}
#endif
