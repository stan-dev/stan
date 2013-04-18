#ifndef __STAN__AGRAD__FWD__COS__HPP__
#define __STAN__AGRAD__FWD__COS__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    cos(const fvar<T>& x) {
      using std::sin;
      using std::cos;
      return fvar<T>(cos(x.val_), x.d_ * -sin(x.val_));
    }
  }
}
#endif
