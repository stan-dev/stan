#ifndef __STAN__DIFF__FWD__OPERATOR__EQUAL__HPP__
#define __STAN__DIFF__FWD__OPERATOR__EQUAL__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T1, typename T2>
    inline 
    bool
    operator==(const fvar<T1>& x, const fvar<T2>& y) {
      return x.val_ == y.val_;
    }

    template <typename T1, typename T2>
    inline 
    bool
    operator==(const fvar<T1>& x, const T2& y) {
      return x.val_ == y;
    }

    template <typename T1, typename T2>
    inline 
    bool 
    operator==(const T1& x, const fvar<T2>& y) {
      return x == y.val_;
    }
  }
}
#endif
