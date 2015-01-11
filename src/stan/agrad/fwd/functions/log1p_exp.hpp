#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG1P_EXP_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG1P_EXP_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log1p_exp.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log1p_exp(const fvar<T>& x) {
      using stan::math::log1p_exp;
      using std::exp;
      return fvar<T>(log1p_exp(x.val_), x.d_ / (1 + exp(-x.val_)));
    }
  }
}
#endif
