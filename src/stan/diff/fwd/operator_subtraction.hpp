#ifndef __STAN__DIFF__FWD__OPERATOR_SUBTRACTION_HPP__
#define __STAN__DIFF__FWD__OPERATOR_SUBTRACTION_HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan{

  namespace diff{

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const fvar<T1>& x1, 
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ - x2.val_, 
                                                  x1.d_ - x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const T1& x1,
              const fvar<T2>& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1 - x2.val_, 
                                                  -x2.d_);
    }

    template <typename T1, typename T2>
    inline
    fvar<typename stan::return_type<T1,T2>::type>
    operator-(const fvar<T1>& x1, 
              const T2& x2) {
      return fvar<typename 
                  stan::return_type<T1,T2>::type>(x1.val_ - x2, 
                                                  x1.d_);
    }
  }
}
#endif
