#ifndef __STAN__AGRAD__FWD__POW__HPP__
#define __STAN__AGRAD__FWD__POW__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/agrad/fwd/inv.hpp>
#include <stan/agrad/fwd/inv_sqrt.hpp>
#include <stan/agrad/fwd/inv_square.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    pow(const fvar<T>& x1, const fvar<T>& x2) {
      using std::pow;
      using std::log;
      T pow_x1_x2(pow(x1.val_,x2.val_));
      return fvar<T>(pow_x1_x2,
                       (x2.d_ * log(x1.val_) 
                         + x2.val_ * x1.d_ / x1.val_) * pow_x1_x2);
    }
  }
}
#endif
