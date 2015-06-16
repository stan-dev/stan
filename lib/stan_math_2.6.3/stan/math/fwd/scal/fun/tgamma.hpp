#ifndef STAN_MATH_FWD_SCAL_FUN_TGAMMA_HPP
#define STAN_MATH_FWD_SCAL_FUN_TGAMMA_HPP

#include <stan/math/fwd/core.hpp>

#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    tgamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::tgamma;
      T u = tgamma(x.val_);
      return fvar<T>(u, x.d_ * u * digamma(x.val_));
    }
  }
}
#endif
