#ifndef STAN__MATH__FWD__SCAL__FUN__LMGAMMA_HPP
#define STAN__MATH__FWD__SCAL__FUN__LMGAMMA_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/lmgamma.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<typename stan::return_type<T,int>::type>
    lmgamma(int x1, const fvar<T>& x2) {
      using stan::math::lmgamma;
      using stan::math::digamma;
      using std::log;
      T deriv = 0;
      for(int count = 1; count < x1 + 1; count++)
        deriv += x2.d_ * digamma(x2.val_ + (1.0 - count) / 2.0);
      return fvar<typename 
                stan::return_type<T,int>::type>(lmgamma(x1, x2.val_), deriv);
    }
  }
}
#endif
