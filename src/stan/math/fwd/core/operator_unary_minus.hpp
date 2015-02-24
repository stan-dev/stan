#ifndef STAN__MATH__FWD__CORE__OPERATOR_UNARY_MINUS_HPP
#define STAN__MATH__FWD__CORE__OPERATOR_UNARY_MINUS_HPP

#include <stan/math/fwd/core.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline 
    fvar<T>
    operator-(const fvar<T>& x) {
      return fvar<T>(-x.val_, -x.d_);
    }
  }
}
#endif
