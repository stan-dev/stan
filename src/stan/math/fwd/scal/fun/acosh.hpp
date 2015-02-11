#ifndef STAN__AGRAD__FWD__FUNCTIONS__ACOSH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ACOSH_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <math.h>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    acosh(const fvar<T>& x) {
      using ::acosh;
      using stan::math::square;
      using std::sqrt;
      using stan::math::NOT_A_NUMBER;
      return fvar<T>(acosh(x.val_),
                     x.d_ / sqrt(square(x.val_) - 1));
    }
  }
}
#endif
