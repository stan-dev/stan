#ifndef STAN__AGRAD__FWD__FUNCTIONS__BINARY_LOG_LOSS_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__BINARY_LOG_LOSS_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/binary_log_loss.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    binary_log_loss(const int y, const fvar<T>& y_hat) {
      using stan::math::binary_log_loss;

      if (y)
        return fvar<T>(binary_log_loss(y,y_hat.val_), -y_hat.d_ / y_hat.val_);
      else
        return fvar<T>(binary_log_loss(y,y_hat.val_), y_hat.d_ / (1.0 - y_hat.val_));
    }
  }
}
#endif
