#ifndef STAN__AGRAD__REV__FUNCTIONS__IS_INF_HPP
#define STAN__AGRAD__REV__FUNCTIONS__IS_INF_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/internal/v_vari.hpp>
#include <stan/math/functions/is_inf.hpp>
#include <stan/math/constants.hpp>

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
