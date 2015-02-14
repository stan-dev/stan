#ifndef STAN__MATH__FWD__SCAL__FUN__OPERATOR_NOT_EQUAL_HPP
#define STAN__MATH__FWD__SCAL__FUN__OPERATOR_NOT_EQUAL_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline 
    bool
    operator!=(const fvar<T>& x, const fvar<T>& y) {
      return x.val_ != y.val_;
    }

    template <typename T>
    inline 
    bool
    operator!=(const fvar<T>& x, double y) {
      return x.val_ != y;
    }

    template <typename T>
    inline 
    bool
    operator!=(double x, const fvar<T>& y) {
      return x != y.val_;
    }
  }
}
#endif
