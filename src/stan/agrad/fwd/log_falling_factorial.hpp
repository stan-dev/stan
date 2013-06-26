#ifndef __STAN__AGRAD__FWD__LOG_FALLING_FACTORIAL__HPP__
#define __STAN__AGRAD__FWD__LOG_FALLING_FACTORIAL__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/log_falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline fvar<T>
    log_falling_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<T>(log_falling_factorial(x.val_, n.val_), digamma(x.val_ + 1) 
                     * x.d_ - digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<typename stan::return_type<T,double>::type>
    log_falling_factorial(double x, const fvar<T>& n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<typename stan::return_type<T,double>::type>(
        log_falling_factorial(x, n.val_), -digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<typename stan::return_type<T,double>::type>
    log_falling_factorial(const fvar<T>& x, double n) {
      using stan::math::log_falling_factorial;
      using boost::math::digamma;

      return fvar<typename stan::return_type<T,double>::type>(
        log_falling_factorial(x.val_, n), digamma(x.val_ + 1) 
          * x.d_);
    }
  }
}
#endif
