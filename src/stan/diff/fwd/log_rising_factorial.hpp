#ifndef __STAN__DIFF__FWD__LOG_RISING_FACTORIAL__HPP__
#define __STAN__DIFF__FWD__log_RISING_FACTORIAL__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace diff {

    template<typename T>
    inline
    fvar<T>
    log_rising_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<T>(log_rising_factorial(x.val_,n.val_), (digamma(x.val_ 
                          + n.val_) * (x.d_ + n.d_) - digamma(x.val_) * x.d_));
    }

    template<typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    log_rising_factorial(const fvar<T>& x, double n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<typename stan::return_type<T,double>::type>(
        log_rising_factorial(x.val_,n), (digamma(x.val_ + n) 
                                - digamma(x.val_)) * x.d_);
    }

    template<typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    log_rising_factorial(double x, const fvar<T>& n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<typename stan::return_type<T,double>::type>(
        log_rising_factorial(x,n.val_), (digamma(x + n.val_) * n.d_));
    }
  }
}
#endif
