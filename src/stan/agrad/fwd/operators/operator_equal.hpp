#ifndef STAN__AGRAD__FWD__OPERATORS__OPERATOR_EQUAL_HPP
#define STAN__AGRAD__FWD__OPERATORS__OPERATOR_EQUAL_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline 
    bool
    operator==(const fvar<T>& x, const fvar<T>& y) {
      return x.val_ == y.val_;
    }

    template <typename T>
    inline 
    bool
    operator==(const fvar<T>& x, double y) {
      return x.val_ == y;
    }

    template <typename T>
    inline 
    bool 
    operator==(double x, const fvar<T>& y) {
      return x == y.val_;
    }
  }
}
#endif
