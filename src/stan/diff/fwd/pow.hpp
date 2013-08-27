#ifndef __STAN__DIFF__FWD__POW__HPP__
#define __STAN__DIFF__FWD__POW__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/diff/fwd/inv.hpp>
#include <stan/diff/fwd/inv_sqrt.hpp>
#include <stan/diff/fwd/inv_square.hpp>

namespace stan{

  namespace diff{

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
