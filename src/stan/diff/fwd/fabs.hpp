#ifndef __STAN__DIFF__FWD__FABS__HPP__
#define __STAN__DIFF__FWD__FABS__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace diff{

    template<typename T>
    inline
    fvar<T>
    fabs(const fvar<T>& x) {
      using std::fabs;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ > 0.0)
         return fvar<T>(fabs(x.val_), x.d_);
      else if(x.val_ == 0.0)
        return fvar<T>(fabs(x.val_), NOT_A_NUMBER);
      else 
        return fvar<T>(fabs(x.val_), -x.d_);
    }
  }
}
#endif
