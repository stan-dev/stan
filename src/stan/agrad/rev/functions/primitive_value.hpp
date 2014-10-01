#ifndef STAN__AGRAD__REV__FUNCTIONS__PRIMITIVE_VALUE_HPP
#define STAN__AGRAD__REV__FUNCTIONS__PRIMITIVE_VALUE_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/math/functions/primitive_value.hpp>

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
