#ifndef STAN__MATH__FWD__SCAL__FUN__LGAMMA_HPP
#define STAN__MATH__FWD__SCAL__FUN__LGAMMA_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

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
