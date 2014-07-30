#ifndef STAN__AGRAD__FWD__FUNCTIONS__FABS_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FABS_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline
    fvar<T>
    fabs(const fvar<T>& x) {
      using stan::math::NOT_A_NUMBER;
      if (x.val_ > 0.0)
        return x;
      else if (x.val_ < 0.0)
        return fvar<T>(-x.val_, -x.d_);
      else if (x.val_ == 0.0)
        return fvar<T>(0, 0);
      else
        return fvar<T>(NOT_A_NUMBER,NOT_A_NUMBER);
    }
  }
}
#endif
