#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOGIT_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOGIT_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/logit.hpp>
#include <stan/math/functions/square.hpp>
#include <stan/math/constants.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    logit(const fvar<T>& x) {
      using stan::math::logit;
      using stan::math::square;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ > 1 || x.val_ < 0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else 
        return fvar<T>(logit(x.val_), x.d_ / (x.val_ - square(x.val_)));
    }
  }
}
#endif
