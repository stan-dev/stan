#ifndef STAN__MATH__FWD__SCAL__FUN__IS_NAN_HPP
#define STAN__MATH__FWD__SCAL__FUN__IS_NAN_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/fun/is_nan.hpp>

namespace stan {

  namespace agrad {

    /**
     * Returns 1 if the input's value is NaN and 0 otherwise.
     *
     * Delegates to <code>stan::math::is_nan</code>.
     *
     * @param x Value to test.
     * @return <code>1</code> if the value is NaN and <code>0</code> otherwise.
     */
    template <typename T>
    inline 
    int
    is_nan(const fvar<T>& x) {
      using stan::math::is_nan;
      return is_nan(x.val());
    }

  }
}

#endif
