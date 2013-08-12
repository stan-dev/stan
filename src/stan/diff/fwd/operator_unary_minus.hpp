#ifndef __STAN__DIFF__FWD__OPERATOR__UNARY__NEGATIVE__HPP__
#define __STAN__DIFF__FWD__OPERATOR__UNARY__NEGATIVE__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline 
    fvar<T>
    operator-(const fvar<T>& x) {
      return fvar<T>(-x.val_, -x.d_);
    }
  }
}
#endif
