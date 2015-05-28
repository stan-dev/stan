#ifndef STAN_MATH_FWD_SCAL_FUN_LOG1M_EXP_HPP
#define STAN_MATH_FWD_SCAL_FUN_LOG1M_EXP_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/expm1.hpp>
#include <stan/math/prim/scal/fun/log1m_exp.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <cmath>

namespace stan {
  namespace math {

    template <typename T>
    inline
    fvar<T>
    log1m_exp(const fvar<T>& x) {
      using stan::math::log1m_exp;
      using stan::math::NOT_A_NUMBER;
      using ::expm1;
      if (x.val_ >= 0)
        return fvar<T>(NOT_A_NUMBER);
      return fvar<T>(log1m_exp(x.val_), x.d_ / -expm1(-x.val_));
    }

  }
}
#endif
