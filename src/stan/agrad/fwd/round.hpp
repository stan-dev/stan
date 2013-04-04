#ifndef __STAN__AGRAD__FWD__ROUND__HPP__
#define __STAN__AGRAD__FWD__ROUND__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    round(const fvar<T>& x) {
      using boost::math::round;
        return fvar<T>(round(x.val_), 0);
    }
  }
}
#endif
