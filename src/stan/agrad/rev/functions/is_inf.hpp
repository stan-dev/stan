#ifndef STAN_AGRAD_REV_FUNCTIONS_IS_INF_HPP
#define STAN_AGRAD_REV_FUNCTIONS_IS_INF_HPP

#include <stan/math/funtions/is_inf.hpp>

namespace stan {

  namespace agrad {

    /**
     * Returns 1 if the input's value is infinite and 0 otherwise.
     *
     * Delegates to <code>stan::math::is_inf</code>.
     *
     * @param x Value to test.
     * @return <code>1</code> if the value is infinite and <code>0</code> otherwise.
     */
    inline 
    int
    is_inf(const var& v) {
      return stan::math::is_inf(v.val());
    }

  }
}

#endif
