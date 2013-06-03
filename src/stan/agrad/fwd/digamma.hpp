#ifndef __STAN__AGRAD__FWD__DIGAMMA__HPP__
#define __STAN__AGRAD__FWD__DIGAMMA__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/zeta.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    digamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::zeta;
      T u = digamma(x.val_);
      return fvar<T>(u, x.d_ * (zeta(2.0) - (0.57721566490153286 + u)));
    }
  }
}
#endif
