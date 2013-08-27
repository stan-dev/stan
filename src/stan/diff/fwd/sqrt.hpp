#ifndef __STAN__DIFF__FWD__SQRT__HPP__
#define __STAN__DIFF__FWD__SQRT__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline 
    fvar<T>
    sqrt(const fvar<T>& x) {
      using std::sqrt;
      return fvar<T>(sqrt(x.val_), x.d_ / (2 * sqrt(x.val_)));
    }
  }
}
#endif
