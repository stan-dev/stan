#ifndef __STAN__AGRAD__FWD__ATAN2__HPP__
#define __STAN__AGRAD__FWD__ATAN2__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    atan2(const fvar<T1>& x1, const fvar<T2>& x2){
      using std::atan2;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(atan2(x1.val_, x2.val_), 
                              (x1.d_ * x2.val_ - x1.val_ * x2.d_) / 
                                 (x2.val_ * x2.val_ + x1.val_ * x1.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    atan2(const T1& x1, const fvar<T2>& x2){
      using std::atan2;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(atan2(x1, x2.val_), 
                     (-x1 * x2.d_) / (x1 * x1 + x2.val_ * x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    atan2(const fvar<T1>& x1, const T2& x2){
      using std::atan2;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(atan2(x1.val_, x2), 
                     (x1.d_ * x2) / (x2 * x2 + x1.val_ * x1.val_));
    }
  }
}
#endif
