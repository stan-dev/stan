#ifndef __STAN__AGRAD__FWD__ATANH__HPP__
#define __STAN__AGRAD__FWD__ATANH__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/atanh.hpp>

namespace stan{

  namespace agrad{

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
