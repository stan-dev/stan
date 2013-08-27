#ifndef __STAN__DIFF__FWD__EXP__HPP__
#define __STAN__DIFF__FWD__EXP__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{


    template <typename T>
    inline
    fvar<T>
    exp(const fvar<T>& x) {
      using std::exp;
      return fvar<T>(exp(x.val_), x.d_ * exp(x.val_));
    }
  }
}
#endif
