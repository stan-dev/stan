#ifndef __STAN__DIFF__FWD__LOG10__HPP__
#define __STAN__DIFF__FWD__LOG10__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    log10(const fvar<T>& x) {
      using std::log;
      using std::log10;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
        return fvar<T>(log10(x.val_), x.d_ / (x.val_ * stan::math::LOG_10));
    }
  }
}
#endif
