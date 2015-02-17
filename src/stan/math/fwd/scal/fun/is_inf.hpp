#ifndef STAN__MATH__FWD__SCAL__FUN__IS_INF_HPP
#define STAN__MATH__FWD__SCAL__FUN__IS_INF_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/fun/is_inf.hpp>

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
    template <typename T>
    inline 
    int
    is_inf(const fvar<T>& x) {
      using stan::math::is_inf;
      return is_inf(x.val());
    }

  }
}

#endif
