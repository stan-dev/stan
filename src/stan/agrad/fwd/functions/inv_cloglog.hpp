#ifndef STAN__AGRAD__FWD__FUNCTIONS__INV_CLOGLOG_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__INV_CLOGLOG_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv_cloglog.hpp>

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
