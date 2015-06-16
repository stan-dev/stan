#ifndef STAN_MATH_FWD_SCAL_FUN_LOG1M_INV_LOGIT_HPP
#define STAN_MATH_FWD_SCAL_FUN_LOG1M_INV_LOGIT_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/log1m_inv_logit.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    log1m_inv_logit(const fvar<T>& x) {
      using std::exp;
      using stan::math::log1m_inv_logit;
      return fvar<T>(log1m_inv_logit(x.val_),
                     -x.d_ / (1 + exp(-x.val_)));
    }
  }
}
#endif
