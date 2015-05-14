#ifndef STAN_MATH_REV_SCAL_FUN_VALUE_OF_REC_HPP
#define STAN_MATH_REV_SCAL_FUN_VALUE_OF_REC_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace math {

    /**
     * Return the value of the specified variable.
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of_rec(const var& v) {
      return v.vi_->val_;
    }

  }
}
#endif
