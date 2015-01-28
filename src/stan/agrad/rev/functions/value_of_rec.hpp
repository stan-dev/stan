#ifndef STAN__AGRAD__REV__FUNCTIONS__VALUE_OF_REC_HPP
#define STAN__AGRAD__REV__FUNCTIONS__VALUE_OF_REC_HPP

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the value of the specified variable.  
     *
     * <p>This function is used internally by finite-diff functions along with
     * <code>stan::math::value_of(T x)</code> to extract the
     * <code>double</code> value of either a scalar or an arbitrarily nested
     * auto-dif variable.  This function will be called when the argument is a
     * <code>stan::agrad::var</code> even if the function is not referred to by
     * namespace because of argument-dependent lookup.
     *
     * @param v Variable.
     * @return Value of variable.
     */
    inline double value_of_rec(const stan::agrad::var& v) {
      return v.vi_->val_;
    }

  }
}
#endif
