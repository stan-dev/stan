#ifndef STAN__MATH__REV__SCAL__FUN__PRIMITIVE_VALUE_HPP
#define STAN__MATH__REV__SCAL__FUN__PRIMITIVE_VALUE_HPP

#include <stan/math/rev/arr/meta/var.hpp>
#include <stan/math/prim/scal/fun/primitive_value.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the primitive double value for the specified auto-diff
     * variable.
     *
     * @param v input variable.
     * @return value of input.
     */
    inline double primitive_value(const agrad::var& v) {
      return v.val();
    }
    
  }
}
#endif
