#ifndef __STAN__DIFF__FWD__LGAMMA__HPP__
#define __STAN__DIFF__FWD__LGAMMA__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    lgamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::lgamma;
      return fvar<T>(lgamma(x.val_), x.d_ * digamma(x.val_));
    }
  }
}
#endif
