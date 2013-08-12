#ifndef __STAN__DIFF__FWD__ACOS__HPP__
#define __STAN__DIFF__FWD__ACOS__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

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
