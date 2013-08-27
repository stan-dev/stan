#ifndef __STAN__DIFF__FWD__ATAN__HPP__
#define __STAN__DIFF__FWD__ATAN__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    atan(const fvar<T>& x) {
      using std::atan;
      return fvar<T>(atan(x.val_), x.d_ / (1 + x.val_ * x.val_));
    }
  }
}
#endif
