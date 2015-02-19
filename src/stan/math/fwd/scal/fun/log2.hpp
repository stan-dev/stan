#ifndef STAN__MATH__FWD__SCAL__FUN__LOG2_HPP
#define STAN__MATH__FWD__SCAL__FUN__LOG2_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/log2.hpp>


namespace stan {

  namespace agrad {

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
