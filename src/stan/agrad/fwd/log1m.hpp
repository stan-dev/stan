#ifndef __STAN__AGRAD__FWD__LOG1M__HPP__
#define __STAN__AGRAD__FWD__LOG1M__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1m.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    log1m(const fvar<T>& x) {
      using stan::math::log1m;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ > 1.0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else 
        return fvar<T>(log1m(x.val_), -x.d_ / (1 - x.val_));
    }
  }
}
#endif
