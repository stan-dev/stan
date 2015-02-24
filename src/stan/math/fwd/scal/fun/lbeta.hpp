#ifndef STAN__MATH__FWD__SCAL__FUN__LBETA_HPP
#define STAN__MATH__FWD__SCAL__FUN__LBETA_HPP

#include <stan/math/fwd/core.hpp>

#include <boost/math/special_functions/digamma.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    lbeta(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::lbeta;
      using boost::math::digamma;
      return fvar<T>(lbeta(x1.val_, x2.val_), 
                     x1.d_ * digamma(x1.val_) 
                     + x2.d_ * digamma(x2.val_)
                     - (x1.d_ + x2.d_) * digamma(x1.val_ + x2.val_));
    }

    template <typename T>
    inline
    fvar<T>
    lbeta(const double x1, const fvar<T>& x2) {
      using stan::math::lbeta;
      using boost::math::digamma;
      return fvar<T>(lbeta(x1, x2.val_), 
                     x2.d_ * digamma(x2.val_) - x2.d_ * digamma(x1 + x2.val_));
    }

    template <typename T>
    inline
    fvar<T>
    lbeta(const fvar<T>& x1, const double x2) {
      using stan::math::lbeta;
      using boost::math::digamma;
      return fvar<T>(lbeta(x1.val_, x2), 
                     x1.d_ * digamma(x1.val_) - x1.d_ * digamma(x1.val_ + x2));
    }
  }
}
#endif
