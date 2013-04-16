#ifndef __STAN__AGRAD__FWD__LBETA__HPP__
#define __STAN__AGRAD__FWD__LBETA__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/lbeta.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lbeta(const fvar<T1>& x1, const fvar<T2>& x2){
      using stan::math::lbeta;
      using boost::math::tgamma;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lbeta(x1.val_, x2.val_), 
                          x1.d_ / tgamma(x1.val_) 
                        + x2.d_ / tgamma(x2.val_)
                        - (x1.d_ + x2.d_) / tgamma(x1.val_ + x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lbeta(const T1& x1, const fvar<T2>& x2){
      using stan::math::lbeta;
      using boost::math::tgamma;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lbeta(x1, x2.val_), 
                    x2.d_ / tgamma(x2.val_) - x2.d_ / tgamma(x1 + x2.val_));
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lbeta(const fvar<T1>& x1, const T2& x2){
      using stan::math::lbeta;
      using boost::math::tgamma;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lbeta(x1.val_, x2), 
                          x1.d_ / tgamma(x1.val_) 
                        - x1.d_ / tgamma(x1.val_ + x2));
    }
  }
}
#endif
