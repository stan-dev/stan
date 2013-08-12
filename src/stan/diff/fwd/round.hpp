#ifndef __STAN__DIFF__FWD__ROUND__HPP__
#define __STAN__DIFF__FWD__ROUND__HPP__

#include <boost/math/special_functions/round.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace diff {

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
