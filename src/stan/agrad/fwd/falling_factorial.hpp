#ifndef __STAN__AGRAD__FWD__FALLING_FACTORIAL__HPP__
#define __STAN__AGRAD__FWD__FALLING_FACTORIAL__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/falling_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace agrad {

    template<typename T1, typename T2>
    inline fvar<typename stan::return_type<T1,T2>::type>
    falling_factorial(const fvar<T1>& x, const fvar<T2>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      typename stan::return_type<T1,T2>::type falling_fact(
                                          falling_factorial(x.val_,n.val_));
      return fvar<typename stan::return_type<T1,T2>::type>(falling_fact, 
        falling_fact, falling_fact * digamma(x.val_ + 1) * x.d_ 
          - falling_fact * digamma(n.val_ + 1) * n.d_);
    }

    template<typename T1, typename T2>
    inline fvar<typename stan::return_type<T1,T2>::type>
    falling_factorial(const fvar<T1>& x, T2 n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      typename stan::return_type<T1,T2>::type falling_fact(
                                                 falling_factorial(x.val_,n));
      return fvar<typename stan::return_type<T1,T2>::type>(falling_fact, 
        falling_fact * digamma(x.val_ + 1) * x.d_);
    }

    template<typename T1, typename T2>
    inline fvar<typename stan::return_type<T1,T2>::type>
    falling_factorial(T1 x, const fvar<T2>& n) {
      using stan::math::falling_factorial;
      using boost::math::digamma;

      typename stan::return_type<T1,T2>::type falling_fact(falling_factorial(x,
                                                                    n.val_));
      return fvar<typename stan::return_type<T1,T2>::type>(falling_fact, 
        -falling_fact * digamma(n.val_ + 1) * n.d_);
    }
  }
}
#endif
