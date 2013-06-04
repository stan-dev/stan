#ifndef __STAN__AGRAD__FWD__BINARY__LOG__LOSS__HPP__
#define __STAN__AGRAD__FWD__BINARY__LOG__LOSS__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/binary_log_loss.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    binary_log_loss(int x1, const fvar<T>& x2) {
      using stan::math::binary_log_loss;

      if(x1 == 1)
        return fvar<T>(binary_log_loss(x1,x2.val_), -x2.d_ / x2.val_);
      if(x1 == 0)
        return fvar<T>(binary_log_loss(x1,x2.val_), x2.d_ / x2.val_);
    }
  }
}
#endif
