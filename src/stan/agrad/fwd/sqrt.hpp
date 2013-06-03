#ifndef __STAN__AGRAD__FWD__SQRT__HPP__
#define __STAN__AGRAD__FWD__SQRT__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv_sqrt.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline 
    fvar<T>
    sqrt(const fvar<T>& x) {
      using std::sqrt;
      using stan::math::inv_sqrt;
      return fvar<T>(sqrt(x.val_), 0.5 * x.d_ * inv_sqrt(x.val_));
    }
  }
}
#endif
