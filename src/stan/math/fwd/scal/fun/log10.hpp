#ifndef STAN__MATH__FWD__SCAL__FUN__LOG10_HPP
#define STAN__MATH__FWD__SCAL__FUN__LOG10_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

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
