#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG1M_INV_LOGIT_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG1M_INV_LOGIT_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1m_inv_logit.hpp>

namespace stan {

  namespace agrad {

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
