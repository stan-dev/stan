#ifndef __STAN__DIFF__FWD__INV_SQRT_HPP__
#define __STAN__DIFF__FWD__INV_SQRT_HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {

  namespace diff {

    template <typename T>
    inline
    fvar<T>
    inv_sqrt(const fvar<T>& x) {
      using std::sqrt;
      T sqrt_x(sqrt(x.val_));
      return fvar<T>(1 / sqrt_x, -0.5 * x.d_ / (x.val_ * sqrt_x));
    }
  }
}
#endif
