#ifndef STAN__MATH__REV__SCAL__FUN__VALUE_OF_HPP
#define STAN__MATH__REV__SCAL__FUN__VALUE_OF_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.
     *
     * <p>This function is used internally by auto-dif functions along
     * with <code>stan::math::value_of(T x)</code> to extract the
     * <code>double</code> value of either a scalar or an auto-dif
     * variable.  This function will be called when the argument is a
     * <code>stan::agrad::var</code> even if the function is not
     * referred to by namespace because of argument-dependent lookup.
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of(const agrad::var& v) {
      return v.vi_->val_;
    }

  }
}
#endif
