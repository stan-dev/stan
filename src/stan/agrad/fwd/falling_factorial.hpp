#ifndef __STAN__AGRAD__FWD__FALLING_FACTORIAL__HPP__
#define __STAN__AGRAD__FWD__FALLING_FACTORIAL__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template<typename T>
    inline fvar<T>
    falling_factorial(const fvar<T>& x, const fvar<T>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      T falling_fact(falling_factorial(x.val_,n.val_));
      return fvar<T>(falling_fact, falling_fact * digamma(x.val_ + 1) * x.d_ 
                     - falling_fact * digamma(n.val_ + 1) * n.d_);
    }

    template<typename T>
    inline fvar<typename stan::return_type<T,double>::type>
    falling_factorial(const fvar<T>& x, double n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      typename boost::math::tools::promote_args<T,double>::type falling_fact(
                                                 falling_factorial(x.val_,n));
      return fvar<typename stan::return_type<T,double>::type>(falling_fact, 
        falling_fact * digamma(x.val_ + 1) * x.d_);
    }

    template<typename T>
    inline fvar<typename stan::return_type<T,double>::type>
    falling_factorial(double x, const fvar<T>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      typename boost::math::tools::promote_args<T,double>::type 
        falling_fact(falling_factorial(x, n.val_));
      return fvar<typename stan::return_type<T,double>::type>(falling_fact, 
        -falling_fact * digamma(n.val_ + 1) * n.d_);
    }
  }
}
#endif
