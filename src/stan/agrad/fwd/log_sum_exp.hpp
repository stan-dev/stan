#ifndef __STAN__AGRAD__FWD__LOG__SUM__EXP__HPP__
#define __STAN__AGRAD__FWD__LOG__SUM__EXP__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_sum_exp.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    log_sum_exp(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<T>(log_sum_exp(x1.val_, x2.val_),
                     x1.d_ / (1 + exp(x2.val_ - x1.val_))
                   + x2.d_ / (exp(x1.val_ - x2.val_) + 1));
    }

    template <typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    log_sum_exp(double x1, const fvar<T>& x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<typename 
                  stan::return_type<T,double>::type>(log_sum_exp(x1, x2.val_),
                          x2.d_ / (exp(x1 - x2.val_) + 1));
    }

    template <typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    log_sum_exp(const fvar<T>& x1, double x2) {
      using stan::math::log_sum_exp;
      using std::exp;
      return fvar<typename 
                  stan::return_type<T,double>::type>(log_sum_exp(x1.val_, x2),
                          x1.d_ / (1 + exp(x2 - x1.val_)));
    }
  }
}
#endif
