#ifndef __STAN__DIFF__FWD__ATANH__HPP__
#define __STAN__DIFF__FWD__ATANH__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/atanh.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline
    fvar<T>
    atanh(const fvar<T>& x) {
      using boost::math::atanh;
       return fvar<T>(atanh(x.val_), x.d_ / (1 - x.val_ * x.val_));
    }
  }
}
#endif
