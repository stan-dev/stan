#ifndef STAN__AGRAD__FWD__FUNCTIONS__TGAMMA_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__TGAMMA_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
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
