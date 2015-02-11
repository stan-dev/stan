#ifndef STAN__AGRAD__FWD__FUNCTIONS__EXP_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__EXP_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <math.h>

namespace stan {

  namespace agrad {


    template <typename T>
    inline
    fvar<T>
    exp(const fvar<T>& x) {
      using ::exp;
      return fvar<T>(exp(x.val_), x.d_ * exp(x.val_));
    }
  }
}
#endif
