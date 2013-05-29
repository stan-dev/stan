#ifndef __STAN__AGRAD__FWD__RISING_FACTORIAL__HPP__
#define __STAN__AGRAD__FWD__RISING_FACTORIAL__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/rising_factorial.hpp>
#include <boost/math/special_functions/digamma.hpp>

namespace stan{

  namespace agrad{

    template <typename T>
    inline
    fvar<T>
    rising_factorial(const fvar<T> x, const int & n) {
      using stan::math::rising_factorial;
      using boost::math::digamma;

      T rising_fact(rising_factorial(x.val_,n));
      return fvar<T>(rising_fact, rising_fact
                     * x.d_ * (digamma(x.val_ + n) - digamma(x.val_)));
    }
  }
}
#endif
