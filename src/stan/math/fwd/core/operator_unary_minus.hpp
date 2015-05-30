#ifndef STAN_MATH_FWD_CORE_OPERATOR_UNARY_MINUS_HPP
#define STAN_MATH_FWD_CORE_OPERATOR_UNARY_MINUS_HPP

#include <stan/math/fwd/core/fvar.hpp>


namespace stan {

  namespace math {

    template <typename T>
    inline
    fvar<T>
    operator-(const fvar<T>& x) {
      return fvar<T>(-x.val_, -x.d_);
    }
  }
}
#endif
