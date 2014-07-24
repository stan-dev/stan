#ifndef STAN__AGRAD__FWD__FUNCTIONS__FMA_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FMA_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/fma.hpp>

namespace stan {

  namespace agrad {

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const fvar<T2>& x2, const fvar<T3>& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1.val_, x2.val_, 
                      x3.val_), x1.d_ * x2.val_ + x2.d_ * x1.val_ + x3.d_);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const fvar<T2>& x2, const fvar<T3>& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1, x2.val_, x3.val_),
                                                     x2.d_ * x1 + x3.d_);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const T2& x2, const fvar<T3>& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1.val_, x2, x3.val_),
                                                     x1.d_ * x2 + x3.d_);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const fvar<T2>& x2, const T3& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1.val_, x2.val_, x3),
                                            x1.d_ * x2.val_ + x2.d_ * x1.val_);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const T2& x2, const fvar<T3>& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1, x2, x3.val_), 
                        x3.d_);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const fvar<T1>& x1, const T2& x2, const T3& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1.val_, x2, x3), 
                        x1.d_ * x2);
    }

    template <typename T1, typename T2, typename T3>
    inline
    fvar<typename stan::return_type<T1,T2,T3>::type>
    fma(const T1& x1, const fvar<T2>& x2, const T3& x3) {
      using stan::math::fma;
      return fvar<typename 
                  stan::return_type<T1,T2,T3>::type>(fma(x1, x2.val_, x3), 
                        x2.d_ * x1);
    }
  }
}
#endif
