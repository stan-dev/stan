#ifndef __STAN__AGRAD__FWD__ASINH__HPP__
#define __STAN__AGRAD__FWD__ASINH__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/asinh.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    asinh(const fvar<T>& x) {
      using boost::math::asinh;
      using std::sqrt;
      return fvar<T>(asinh(x.val_), x.d_ / sqrt(x.val_ * x.val_ + 1));
    }
  }
}
#endif
