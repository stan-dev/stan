#ifndef STAN__AGRAD__FWD__FUNCTIONS__INV_CLOGLOG_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__INV_CLOGLOG_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
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
