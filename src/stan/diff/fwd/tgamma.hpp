#ifndef __STAN__DIFF__FWD__TGAMMA__HPP__
#define __STAN__DIFF__FWD__TGAMMA__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    tgamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::tgamma;
      return fvar<T>(tgamma(x.val_), x.d_ * tgamma(x.val_) * digamma(x.val_));
    }
  }
}
#endif
