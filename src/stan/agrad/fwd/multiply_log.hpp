#ifndef __STAN__AGRAD__FWD__MULTIPLY__LOG__HPP__
#define __STAN__AGRAD__FWD__MULTIPLY__LOG__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/multiply_log.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    multiply_log(const fvar<T1>& x1, const fvar<T2>& x2){
      using stan::math::multiply_log;
      using std::log;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(multiply_log(x1.val_, 
                                                               x2.val_),
                             x1.d_ * log(x2.val_) + x1.val_ * x2.d_ / x2.val_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    multiply_log(const T1& x1, const fvar<T2>& x2){
      using stan::math::multiply_log;
      using std::log;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(multiply_log(x1, x2.val_),
                                 x1 * x2.d_ / x2.val_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    multiply_log(const fvar<T1>& x1, const T2& x2){
      using stan::math::multiply_log;
      using std::log;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(multiply_log(x1.val_, x2),
                                                  log(x2));
    }
  }
}
#endif
