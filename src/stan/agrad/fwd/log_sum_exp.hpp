#ifndef __STAN__AGRAD__FWD__LOG__SUM__EXP__HPP__
#define __STAN__AGRAD__FWD__LOG__SUM__EXP__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_sum_exp.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    log_sum_exp(const fvar<T1>& x1, const fvar<T2>& x2){
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(log_sum_exp(x1.val_, 
                                                              x2.val_),
                     (x1.d_ * exp(x1.val_) 
                       + x2.d_ * exp(x2.val_)) / (exp(x1.val_) + exp(x2.val_)));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    log_sum_exp(const T1& x1, const fvar<T2>& x2){
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(log_sum_exp(x1, x2.val_),
                          x2.d_ * exp(x2.val_) / (exp(x1) + exp(x2.val_)));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    log_sum_exp(const fvar<T1>& x1, const T2& x2){
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(log_sum_exp(x1.val_, x2),
                          x1.d_ * exp(x1.val_) / (exp(x1.val_) + exp(x2)));
    }
  }
}
#endif
