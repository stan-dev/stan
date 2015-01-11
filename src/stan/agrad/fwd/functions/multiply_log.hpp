#ifndef STAN__AGRAD__FWD__FUNCTIONS__MULTIPLY_LOG_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__MULTIPLY_LOG_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/multiply_log.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    multiply_log(const fvar<T>& x1, const fvar<T>& x2) {
      using stan::math::multiply_log;
      using std::log;
      return fvar<T>(multiply_log(x1.val_, x2.val_),
                     x1.d_ * log(x2.val_) + x1.val_ * x2.d_ / x2.val_);
    }

    template <typename T>
    inline
    fvar<T>
    multiply_log(const double x1, const fvar<T>& x2) {
      using stan::math::multiply_log;
      using std::log;
      return fvar<T>(multiply_log(x1, x2.val_),
                     x1 * x2.d_ / x2.val_);
    }

    template <typename T>
    inline
    fvar<T>
    multiply_log(const fvar<T>& x1, const double x2) {
      using stan::math::multiply_log;
      using std::log;
      return fvar<T>(multiply_log(x1.val_, x2), 
                     x1.d_ * log(x2));
    }
  }
}
#endif
