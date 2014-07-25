#ifndef STAN__AGRAD__FWD__FUNCTIONS__INV_LOGIT_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__INV_LOGIT_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv_logit.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    inv_logit(const fvar<T>& x) {
      using std::exp;
      using std::pow;
      using stan::math::inv_logit;
      return fvar<T>(inv_logit(x.val_), 
           x.d_ * inv_logit(x.val_) * (1 - inv_logit(x.val_)));
    }
  }
}
#endif
