#ifndef STAN__AGRAD__FWD__FUNCTIONS__ACOSH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ACOSH_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/acosh.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    acosh(const fvar<T>& x) {
      using boost::math::acosh;
      using stan::math::square;
      using std::sqrt;
      using stan::math::NOT_A_NUMBER;
      if(x.val_ < 1)
        return fvar<T>(NOT_A_NUMBER, NOT_A_NUMBER);
      else 
        return fvar<T>(acosh(x.val_),
                       x.d_ / sqrt(square(x.val_) - 1));
    }
  }
}
#endif
