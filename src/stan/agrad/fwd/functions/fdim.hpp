#ifndef STAN__AGRAD__FWD__FUNCTIONS__FDIM_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FDIM_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/fdim.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    fdim(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1.val_ < x2.val_)
        return fvar<T>(fdim(x1.val_, x2.val_), 0);
      else 
        return fvar<T>(fdim(x1.val_, x2.val_),
                       x1.d_ - x2.d_ * floor(x1.val_ / x2.val_));
    }

    template <typename T>
    inline
    fvar<T>
    fdim(const fvar<T>& x1, const double x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1.val_ < x2)
        return fvar<T>(fdim(x1.val_, x2), 0);
      else 
        return fvar<T>(fdim(x1.val_, x2), x1.d_);              
    }

    template <typename T>
    inline
    fvar<T>
    fdim(const double x1, const fvar<T>& x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1 < x2.val_)
        return fvar<T>(fdim(x1, x2.val_), 0);
      else 
        return fvar<T>(fdim(x1, x2.val_), x2.d_ * -floor(x1 / x2.val_));         
    }
  }
}
#endif
