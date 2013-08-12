#ifndef __STAN__DIFF__FWD__LOG2__HPP__
#define __STAN__DIFF__FWD__LOG2__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/log2.hpp>


namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    log2(const fvar<T>& x) {
      using std::log;
      using stan::math::log2;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 0.0)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else
        return fvar<T>(log2(x.val_), x.d_ / (x.val_ * stan::math::LOG_2));
    }
  }
}
#endif
