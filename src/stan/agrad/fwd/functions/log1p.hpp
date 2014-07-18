#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG1P_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG1P_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/math/constants.hpp>

namespace stan {

  namespace agrad {

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
