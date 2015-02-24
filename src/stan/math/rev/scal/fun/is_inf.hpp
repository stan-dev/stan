#ifndef STAN__MATH__REV__SCAL__FUN__IS_INF_HPP
#define STAN__MATH__REV__SCAL__FUN__IS_INF_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/scal/fun/is_inf.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

    /**
     * Returns 1 if the input's value is infinite and 0 otherwise.
     *
     * Delegates to <code>stan::math::is_inf</code>.
     *
     * @param v Value to test.
     *
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
