#ifndef STAN_MATH_FWD_SCAL_FUN_LOG_DIFF_EXP_HPP
#define STAN_MATH_FWD_SCAL_FUN_LOG_DIFF_EXP_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/log_diff_exp.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace math {

    template <typename T> inline fvar<T>
    log_diff_exp(const fvar<T>& x1, const fvar<T>& x2) {
       using stan::math::log_diff_exp;
       using stan::math::NOT_A_NUMBER;
       using std::exp;
       if (x1.val_ <= x2.val_)
         return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
       return fvar<T>(log_diff_exp(x1.val_, x2.val_),
                      x1.d_ / (1 - exp(x2.val_ - x1.val_))
                      + x2.d_ / (1 - exp(x1.val_ - x2.val_)));
    }

    template <typename T1, typename T2> inline fvar<T2>
    log_diff_exp(const T1& x1, const fvar<T2>& x2) {
      using stan::math::log_diff_exp;
      using stan::math::NOT_A_NUMBER;
      using std::exp;
      if (x1 <= x2.val_)
        return fvar<T2>(NOT_A_NUMBER, NOT_A_NUMBER);
      return fvar<T2>(log_diff_exp(x1, x2.val_),
                     x2.d_ / (1 - exp(x1 - x2.val_)));
    }

    template <typename T1, typename T2> inline fvar<T1>
    log_diff_exp(const fvar<T1>& x1, const T2& x2) {
      using stan::math::log_diff_exp;
      using stan::math::NOT_A_NUMBER;
      using std::exp;
      if (x1.val_ <= x2)
        return fvar<T1>(NOT_A_NUMBER, NOT_A_NUMBER);
      return fvar<T1>(log_diff_exp(x1.val_, x2),
                          x1.d_ / (1 - exp(x2 - x1.val_)));
    }
  }
}
#endif
