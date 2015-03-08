#ifndef STAN__MATH__FWD__SCAL__FUN__INV_LOGIT_HPP
#define STAN__MATH__FWD__SCAL__FUN__INV_LOGIT_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/inv_logit.hpp>

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
