#ifndef STAN__AGRAD__FWD__FUNCTIONS__FMAX_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FMAX_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>
#include <math.h>
#include <stan/meta/likely.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    fmax(const fvar<T>& x1, const fvar<T>& x2) {
      using ::fmax;
      using stan::math::NOT_A_NUMBER;
      if (unlikely(boost::math::isnan(x1.val_))) {
        if (boost::math::isnan(x2.val_))
          return fvar<T>(fmax(x1.val_,x2.val_),NOT_A_NUMBER);
        else
          return fvar<T>(x2.val_, x2.d_);
      } else if (unlikely(boost::math::isnan(x2.val_))) 
        return fvar<T>(x1.val_,x1.d_);
      else if(x1.val_ > x2.val_)
        return fvar<T>(x1.val_, x1.d_);
      else if(x1.val_ == x2.val_)
        return fvar<T>(x1.val_, NOT_A_NUMBER);
      else 
        return fvar<T>(x2.val_, x2.d_);      
    }

    template <typename T>
    inline
    fvar<T>
    fmax(const double x1, const fvar<T>& x2) {
      using ::fmax;
      using stan::math::NOT_A_NUMBER;
      if (unlikely(boost::math::isnan(x1))) {
        if (boost::math::isnan(x2.val_))
          return fvar<T>(fmax(x1,x2.val_),NOT_A_NUMBER);
        else
          return fvar<T>(x2.val_, x2.d_);
      } else if (unlikely(boost::math::isnan(x2.val_))) 
        return fvar<T>(x1,0.0);
      else if(x1 > x2.val_)
        return fvar<T>(x1, 0.0);
      else if(x1 == x2.val_)
        return fvar<T>(x2.val_, NOT_A_NUMBER);
      else 
        return fvar<T>(x2.val_, x2.d_);    
    }

    template <typename T>
    inline
    fvar<T>
    fmax(const fvar<T>& x1, const double x2) {
      using ::fmax;
      using stan::math::NOT_A_NUMBER;
      if (unlikely(boost::math::isnan(x1.val_))) {
        if (boost::math::isnan(x2))
          return fvar<T>(fmax(x1.val_,x2),NOT_A_NUMBER);
        else
          return fvar<T>(x2, 0.0);
      } else if (unlikely(boost::math::isnan(x2))) 
        return fvar<T>(x1.val_,x1.d_);
      else if(x1.val_ > x2)
        return fvar<T>(x1.val_, x1.d_);
      else if(x1.val_ == x2)
        return fvar<T>(x1.val_, NOT_A_NUMBER);
      else 
        return fvar<T>(x2, 0.0);
    }
  }
}
#endif
