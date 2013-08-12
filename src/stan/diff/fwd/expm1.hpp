#ifndef __STAN__AGRAD__FWD__EXPM1__HPP__
#define __STAN__AGRAD__FWD__EXPM1__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/expm1.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    expm1(const fvar<T>& x) {
      using boost::math::expm1;
      using std::exp;
      return fvar<T>(expm1(x.val_), x.d_ * exp(x.val_));
    } 
  }
}
#endif
