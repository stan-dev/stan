#ifndef __STAN__AGRAD__FWD__CBRT__HPP__
#define __STAN__AGRAD__FWD__CBRT__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/cbrt.hpp>

namespace stan{

  namespace agrad{

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
