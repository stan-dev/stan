#ifndef __STAN__AGRAD__FWD__COSH__HPP__
#define __STAN__AGRAD__FWD__COSH__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    cosh(const fvar<T>& x) {
      using std::sinh;
      using std::cosh;
      return fvar<T>(cosh(x.val_), x.d_ * sinh(x.val_));
    }
  }
}
#endif
