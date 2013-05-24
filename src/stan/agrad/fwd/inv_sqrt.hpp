#ifndef __STAN__AGRAD__FWD__INV_SQRT_HPP__
#define __STAN__AGRAD__FWD__INV_SQRT_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    inv_sqrt(const fvar<T>& x) {
      using stan::agrad::sqrt;
      using std::sqrt;
      return fvar<T>(1 / sqrt(x.val_), -0.5 * x.d_ / (x.val_ * sqrt(x.val_)));
    }
  }
}
#endif
