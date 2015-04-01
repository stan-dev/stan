#ifndef STAN__MATH__FWD__SCAL__FUN__LOG_HPP
#define STAN__MATH__FWD__SCAL__FUN__LOG_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log(const fvar<T>& x) {
      using std::log;
      using stan::math::NOT_A_NUMBER;
      if (x.val_ < 0.0)
          return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
          return fvar<T>(log(x.val_), x.d_ / x.val_);
    }
  }
}
#endif
