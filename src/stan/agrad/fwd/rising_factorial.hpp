#ifndef __STAN__AGRAD__FWD__RISING_FACTORIAL__HPP__
#define __STAN__AGRAD__FWD__RISING_FACTORIAL__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline
    fvar<T>
    rising_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      T rising_fact(rising_factorial(x.val_,n.val_));
      return fvar<T>(rising_fact, rising_fact * (digamma(x.val_ + n.val_)
                                  * (x.d_ + n.d_) - digamma(x.val_) * x.d_));
    }

    template<typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    rising_factorial(const fvar<T>& x, double n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      typename boost::math::tools::promote_args<T,double>::type 
        rising_fact(rising_factorial(x.val_,n));
      return fvar<typename stan::return_type<T,double>::type>(rising_fact, 
        rising_fact * x.d_ * (digamma(x.val_ + n) - digamma(x.val_)));
    }

    template<typename T>
    inline
    fvar<typename stan::return_type<T,double>::type>
    rising_factorial(double x, const fvar<T>& n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      typename boost::math::tools::promote_args<T,double>::type 
        rising_fact(rising_factorial(x,n.val_));
      return fvar<typename stan::return_type<T,double>::type>(rising_fact, 
        rising_fact * (digamma(x + n.val_) * n.d_));
    }
  }
}
#endif
