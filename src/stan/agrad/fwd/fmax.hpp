#ifndef __STAN__AGRAD__FWD__FMAX__HPP__
#define __STAN__AGRAD__FWD__FMAX__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/constants.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    fmax(const fvar<T>& x1, const fvar<T>& x2) {
      using std::max;
      using stan::math::NOT_A_NUMBER;
      if(x1.val_ > x2.val_)
        return fvar<typename stan::return_type<T>::type>(
          x1.val_, x1.d_ * 1.0);
      else if(x1.val_ == x2.val_)
        return fvar<typename stan::return_type<T>::type>(
          x1.val_, NOT_A_NUMBER);
      else 
        return fvar<typename stan::return_type<T>::type>(
          x2.val_, x2.d_ * 1.0);      
    }

    template <typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    fmax(double x1, const fvar<T>& x2) {
      using std::max;
      using stan::math::NOT_A_NUMBER;
      if(x1 > x2.val_)
        return fvar<typename stan::return_type<T,double>::type>(
          x1, 0.0);
      else if(x1 == x2.val_)
        return fvar<typename stan::return_type<T,double>::type>(
          x2.val_, NOT_A_NUMBER);
      else 
        return fvar<typename stan::return_type<T,double>::type>(
          x2.val_, x2.d_ * 1.0);    
    }

    template <typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    fmax(const fvar<T>& x1, double x2) {
      using std::max;
      using stan::math::NOT_A_NUMBER;
      if(x1.val_ > x2)
        return fvar<typename stan::return_type<T,double>::type>(
          x1.val_, x1.d_ * 1.0);
      else if(x1.val_ == x2)
        return fvar<typename stan::return_type<T,double>::type>(
         x1.val_, NOT_A_NUMBER);
      else 
        return fvar<typename stan::return_type<T,double>::type>(
          x2, 0.0);
     }
  }
}
#endif
