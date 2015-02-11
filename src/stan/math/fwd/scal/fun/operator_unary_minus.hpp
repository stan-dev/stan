#ifndef STAN__AGRAD__FWD__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP
#define STAN__AGRAD__FWD__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>

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
