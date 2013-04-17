#ifndef __STAN__AGRAD__FWD__ACOSH__HPP__
#define __STAN__AGRAD__FWD__ACOSH__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/acosh.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    acosh(const fvar<T>& x) {
      using boost::math::acosh;
      using std::sqrt;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 1)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else 
        return fvar<T>(acosh(x.val_),
                     x.d_ /(sqrt(x.val_ * x.val_ - 1)));
    }
  }
}
#endif
