#ifndef __STAN__DIFF__FWD__CEIL__HPP__
#define __STAN__DIFF__FWD__CEIL__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    ceil(const fvar<T>& x) {
      using std::ceil;
        return fvar<T>(ceil(x.val_), 0);
    }
  }
}
#endif
