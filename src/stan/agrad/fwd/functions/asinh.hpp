#ifndef STAN__AGRAD__FWD__FUNCTIONS__ASINH_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__ASINH_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/asinh.hpp>
#include <stan/math/functions/square.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    asinh(const fvar<T>& x) {
      using boost::math::asinh;
      using std::sqrt;
      using stan::math::square;
      return fvar<T>(asinh(x.val_), x.d_ / sqrt(square(x.val_) + 1));
    }
  }
}
#endif
