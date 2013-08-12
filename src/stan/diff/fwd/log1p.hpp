#ifndef __STAN__DIFF__FWD__LOG1P__HPP__
#define __STAN__DIFF__FWD__LOG1P__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    log1p(const fvar<T>& x) {
      using stan::math::log1p;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < -1.0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else 
        return fvar<T>(log1p(x.val_), x.d_ / (1 + x.val_));
    }
  }
}
#endif
