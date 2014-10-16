#ifndef STAN_AGRAD_REV_FUNCTIONS_IS_NAN_HPP
#define STAN_AGRAD_REV_FUNCTIONS_IS_NAN_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/functions/is_nan.hpp>
#include <stan/math/constants.hpp>

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
    inline 
    int
    is_nan(const var& v) {
      return stan::math::is_nan(v.val());
    }

  }
}

#endif
