#ifndef STAN__MATH__FWD__CORE__OPERATOR_LESS_THAN_OR_EQUAL_HPP
#define STAN__MATH__FWD__CORE__OPERATOR_LESS_THAN_OR_EQUAL_HPP

#include <stan/math/fwd/core/fvar.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    bool
    operator<=(const fvar<T>& x, const fvar<T>& y) {
      return x.val_ <= y.val_;
    }

    template <typename T>
    inline
    bool
    operator<=(const fvar<T>& x, double y) {
      return x.val_ <= y;
    }

    template <typename T>
    inline
    bool
    operator<=(double x, const fvar<T>& y) {
      return x <= y.val_;
    }
  }
}
#endif
