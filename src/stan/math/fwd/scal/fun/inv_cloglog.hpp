#ifndef STAN__MATH__FWD__SCAL__FUN__INV_CLOGLOG_HPP
#define STAN__MATH__FWD__SCAL__FUN__INV_CLOGLOG_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/inv_cloglog.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    inv_cloglog(const fvar<T>& x) {
      using std::exp;
      using stan::math::inv_cloglog;
      return fvar<T>(inv_cloglog(x.val_), x.d_ * exp(x.val_ - exp(x.val_)));
    }
  }
}
#endif
