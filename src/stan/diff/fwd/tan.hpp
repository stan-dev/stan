#ifndef __STAN__DIFF__FWD__TAN__HPP__
#define __STAN__DIFF__FWD__TAN__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

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
