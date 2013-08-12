#ifndef __STAN__AGRAD__FWD__LMGAMMA__HPP__
#define __STAN__AGRAD__FWD__LMGAMMA__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/functions/lmgamma.hpp>

namespace stan{

  namespace agrad{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lmgamma(const fvar<T1>& x1, const fvar<T2>& x2){
      using stan::math::lmgamma;
      using boost::math::digamma;
      using std::log;
      double deriv = 0;
      int count;
      for(count = 1; count < x2.val_ - 1; count++)
        deriv += (x1.d_  - x2.d_ / 2) * digamma(x1.val_ 
                                                - (x2.val_ - count) / 2);
      deriv += x1.d_ * digamma(x1.val_);
      deriv += (2 * x2.val_ - 1) / 2 
        * log(boost::math::constants::pi<double>()) * x2.d_;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lmgamma(x1.val_, x2.val_), 
                                                  deriv);
    }
 
    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lmgamma(const T1& x1, const fvar<T2>& x2){
      using stan::math::lmgamma;
      using boost::math::digamma;
      using std::log;
      double deriv = 0;
      int count;
      for(count = 1; count < x2.val_ - 1; count++)
        deriv += (0  - x2.d_ / 2) * digamma(x1 - (x2.val_ - count) / 2);
      deriv += 0 * digamma(x1);
      deriv += (2 * x2.val_ - 1) / 2 
        * log(boost::math::constants::pi<double>()) * x2.d_;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lmgamma(x1, x2.val_), deriv);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    lmgamma(const fvar<T1>& x1, const T2& x2){
      using stan::math::lmgamma;
      using boost::math::digamma;
      using std::log;
      double deriv = 0;
      int count;
      for(count = 1; count < x2 - 1; count++)
        deriv += (x1.d_  - 0) * digamma(x1.val_ - (x2 - count) / 2);
      deriv += x1.d_ * digamma(x1.val_);
      deriv += (2 * x2 - 1) / 2 
        * log(boost::math::constants::pi<double>()) * 0;
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(lmgamma(x1.val_, x2), deriv);
    }
  }
}
#endif
