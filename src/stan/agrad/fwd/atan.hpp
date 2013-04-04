#ifndef __STAN__AGRAD__FWD__ATAN__HPP__
#define __STAN__AGRAD__FWD__ATAN__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    atan(const fvar<T>& x) {
      using std::atan;
      return fvar<T>(atan(x.val_), x.d_ / (1 + x.val_ * x.val_));
    }
  }
}
#endif
