#ifndef STAN__AGRAD__FWD__FUNCTIONS__LGAMMA_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LGAMMA_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
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
