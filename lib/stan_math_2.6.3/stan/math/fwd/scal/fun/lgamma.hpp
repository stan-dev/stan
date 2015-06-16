#ifndef STAN_MATH_FWD_SCAL_FUN_LGAMMA_HPP
#define STAN_MATH_FWD_SCAL_FUN_LGAMMA_HPP

#include <stan/math/fwd/core.hpp>

#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    lgamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::lgamma;
      return fvar<T>(lgamma(x.val_), x.d_ * digamma(x.val_));
    }
  }
}
#endif
