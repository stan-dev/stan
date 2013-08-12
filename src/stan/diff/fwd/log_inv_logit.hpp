#ifndef __STAN__DIFF__FWD__LOG__INV__LOGIT__HPP__
#define __STAN__DIFF__FWD__LOG__INV__LOGIT__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_inv_logit.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    log_inv_logit(const fvar<T>& x) {
      using std::exp;
      using stan::math::log_inv_logit;
      return fvar<T>(log_inv_logit(x.val_),
                        x.d_ * exp(-x.val_) / (1 + exp(-x.val_))); 
    }
  }
}
#endif
