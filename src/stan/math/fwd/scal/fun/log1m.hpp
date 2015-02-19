#ifndef STAN__MATH__FWD__SCAL__FUN__LOG1M_HPP
#define STAN__MATH__FWD__SCAL__FUN__LOG1M_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

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
