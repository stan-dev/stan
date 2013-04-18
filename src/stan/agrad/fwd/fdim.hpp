#ifndef __STAN__AGRAD__FWD__FDIM__HPP__
#define __STAN__AGRAD__FWD__FDIM__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/fdim.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    fdim(const fvar<T1>& x1, const fvar<T2>& x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1.val_ < x2.val_)
        return fvar<typename 
                       stan::return_type<T1,T2>::type>(fdim(x1.val_, x2.val_),
                                                       0);
      else 
        return fvar<typename
                    stan::return_type<T1,T2>::type>(fdim(x1.val_, x2.val_),
                           x1.d_ - x2.d_ * floor(x1.val_ / x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    fdim(const fvar<T1>& x1, const T2& x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1.val_ < x2)
           return fvar<typename 
                       stan::return_type<T1,T2>::type>(fdim(x1.val_, x2), 0);
      else 
        return fvar<typename stan::return_type<T1,T2>::type>(fdim(x1.val_, x2),
                                          x1.d_);              
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    fdim(const T1& x1, const fvar<T2>& x2) {
      using stan::math::fdim;
      using std::floor;
      if(x1 < x2.val_)
           return fvar<typename 
                       stan::return_type<T1,T2>::type>(fdim(x1, x2.val_), 0);
      else 
        return fvar<typename stan::return_type<T1,T2>::type>(fdim(x1, x2.val_),
                                          x2.d_ * -floor(x1 / x2.val_));         
    }
  }
}
#endif
