#ifndef __STAN__AGRAD__FWD__OPERATOR__UNARY__NEGATIVE__HPP__
#define __STAN__AGRAD__FWD__OPERATOR__UNARY__NEGATIVE__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline 
    fvar<T>
    operator-(const fvar<T>& x) {
      return fvar<T>(-x.val_, -x.d_);
    }
  }
}
#endif
