#ifndef __STAN__AGRAD__FWD__ABS__HPP__
#define __STAN__AGRAD__FWD__ABS__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace agrad{

    template<typename T>
    inline
    fvar<T>
    abs(const fvar<T>& x) {
      using std::abs;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ > 0.0)
        return fvar<T>(abs(x.val_), x.d_);
      else if(x.val_ == 0.0)
        return fvar<T>(abs(x.val_), NOT_A_NUMBER);
      else 
        return fvar<T>(abs(x.val_), -x.d_);
    }
  }
}
#endif
