#ifndef STAN__MATH__FWD__SCAL__FUN__TGAMMA_HPP
#define STAN__MATH__FWD__SCAL__FUN__TGAMMA_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    tgamma(const fvar<T>& x) {
      using boost::math::digamma;
      using boost::math::tgamma;
      T u = tgamma(x.val_);
      return fvar<T>(u, x.d_ * u * digamma(x.val_));
    }
  }
}
#endif
