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
    binary_log_loss(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::binary_log_loss;
      using std::log;
      return fvar<T>(binary_log_loss(x1.val_, x2.val_),
                                      x1.d_ * log(1 / x2.val_ - 1)
                                      - x2.d_ * x1.val_ / x2.val_ 
                                      + x2.d_ * (1 - x1.val_) / (1 - x2.val_));
    }
  }
}
#endif
