#ifndef __STAN__DIFF__FWD__ASINH__HPP__
#define __STAN__DIFF__FWD__ASINH__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/asinh.hpp>

namespace stan{

  namespace diff{

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
