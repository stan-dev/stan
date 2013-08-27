#ifndef __STAN__DIFF__FWD__ASIN__HPP__
#define __STAN__DIFF__FWD__ASIN__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    asin(const fvar<T>& x) {
      using std::asin;
      using std::sqrt;
      return fvar<T>(asin(x.val_), x.d_ / sqrt(1 - x.val_ * x.val_));
    }
  }
}
#endif
