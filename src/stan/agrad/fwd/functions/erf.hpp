#ifndef STAN__AGRAD__FWD__FUNCTIONS__ERF_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ERF_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    erf(const fvar<T>& x) {
      using boost::math::erf;
      using std::sqrt;
      using std::exp;
      using stan::math::square;
      return fvar<T>(erf(x.val_), x.d_ * exp(-square(x.val_)) 
                                  * stan::math::TWO_OVER_SQRT_PI);
    }
  }
}
#endif
