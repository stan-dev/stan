#ifndef __STAN__DIFF__FWD__TRUNC__HPP__
#define __STAN__DIFF__FWD__TRUNC__HPP__

#include <boost/math/special_functions/trunc.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace diff {

    template <typename T>
    inline
    fvar<T>
    trunc(const fvar<T>& x) {
      using boost::math::trunc;
      return fvar<T>(trunc(x.val_), 0);
    }

  }
}
#endif
