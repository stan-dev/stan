#ifndef __STAN__AGRAD__FWD__INV__LOGIT__HPP__
#define __STAN__AGRAD__FWD__INV__LOGIT__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv_logit.hpp>

namespace stan{

  namespace agrad{

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
