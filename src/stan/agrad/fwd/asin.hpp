#ifndef __STAN__AGRAD__FWD__ASIN__HPP__
#define __STAN__AGRAD__FWD__ASIN__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

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
