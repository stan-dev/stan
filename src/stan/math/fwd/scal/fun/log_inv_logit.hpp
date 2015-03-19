#ifndef STAN__MATH__FWD__SCAL__FUN__LOG_INV_LOGIT_HPP
#define STAN__MATH__FWD__SCAL__FUN__LOG_INV_LOGIT_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/log_inv_logit.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log_inv_logit(const fvar<T>& x) {
      using std::exp;
      using stan::math::log_inv_logit;
      return fvar<T>(log_inv_logit(x.val_),
                        x.d_  / (1 + exp(x.val_)));
    }
  }
}
#endif
