#ifndef STAN__AGRAD__FWD__FUNCTIONS__ATANH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ATANH_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    atanh(const fvar<T>& x) {
      using boost::math::atanh;
      using stan::math::square;
      return fvar<T>(atanh(x.val_), x.d_ / (1 - square(x.val_)));
    }
  }
}
#endif
