#ifndef STAN__AGRAD__FWD__FUNCTIONS__LOG_RISING_FACTORIAL_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__LOG_RISING_FACTORIAL_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline
    fvar<T>
    log_rising_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<T>(log_rising_factorial(x.val_,n.val_), 
                     (digamma(x.val_ + n.val_) * (x.d_ + n.d_) - digamma(x.val_) * x.d_));
    }

    template<typename T>
    inline
    fvar<T>
    log_rising_factorial(const fvar<T>& x, const double n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<T>(log_rising_factorial(x.val_,n),
                     (digamma(x.val_ + n) - digamma(x.val_)) * x.d_);
                      
    }

    template<typename T>
    inline
    fvar<T>
    log_rising_factorial(const double x, const fvar<T>& n) {
      using stan::math::log_rising_factorial;
      using boost::math::digamma;

      return fvar<T>(log_rising_factorial(x,n.val_), 
                     (digamma(x + n.val_) * n.d_));
    }
  }
}
#endif
