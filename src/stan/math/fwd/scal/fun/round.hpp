#ifndef STAN__MATH__FWD__SCAL__FUN__ROUND_HPP
#define STAN__MATH__FWD__SCAL__FUN__ROUND_HPP

#include <math.h>
#include <stan/math/fwd/core/fvar.hpp>


namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    round(const fvar<T>& x) {
      using ::round;
        return fvar<T>(round(x.val_), 0);
    }

  }
}
#endif
