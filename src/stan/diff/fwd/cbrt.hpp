#ifndef __STAN__DIFF__FWD__CBRT__HPP__
#define __STAN__DIFF__FWD__CBRT__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/cbrt.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    cbrt(const fvar<T>& x) {
      using boost::math::cbrt;
      return fvar<T>(cbrt(x.val_),
                     x.d_ / ( cbrt(x.val_) * cbrt(x.val_) * 3.0));
    }
  }
}
#endif
