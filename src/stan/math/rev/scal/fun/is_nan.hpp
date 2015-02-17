#ifndef STAN__MATH__REV__SCAL__FUN__IS_NAN_HPP
#define STAN__MATH__REV__SCAL__FUN__IS_NAN_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/scal/fun/v_vari.hpp>
#include <stan/math/prim/scal/fun/is_nan.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

    /**
     * Returns 1 if the input's value is NaN and 0 otherwise.
     *
     * Delegates to <code>stan::math::is_nan(double)</code>.
     *
     * @param v Value to test.
     *
     * @return <code>1</code> if the value is NaN and <code>0</code> otherwise.
     */
    inline 
    int
    is_nan(const var& v) {
      return stan::math::is_nan(v.val());
    }

  }
}

#endif
