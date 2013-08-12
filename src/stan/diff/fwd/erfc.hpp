#ifndef __STAN__AGRAD__FWD__ERFC__HPP__
#define __STAN__AGRAD__FWD__ERFC__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    erfc(const fvar<T>& x) {
      using boost::math::erfc;
      using std::sqrt;
      using std::exp;
      return fvar<T>(erfc(x.val_), -x.d_ * exp(-x.val_ * x.val_) 
                                    * stan::math::TWO_OVER_SQRT_PI);
    }
  }
}
#endif
