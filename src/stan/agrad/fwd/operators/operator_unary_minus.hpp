#ifndef __STAN__AGRAD__FWD__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP__
#define __STAN__AGRAD__FWD__OPERATORS__OPERATOR_UNARY_NEGATIVE_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
