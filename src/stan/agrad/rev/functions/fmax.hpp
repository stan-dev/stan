#ifndef STAN__AGRAD__REV__FUNCTIONS__FMAX_HPP
#define STAN__AGRAD__REV__FUNCTIONS__FMAX_HPP

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Returns the maximum of the two variable arguments (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * No new variable implementations are created, with this function
     * defined as if by
     *
     * <code>fmax(a,b) = a</code> if a's value is greater than b's, and .
     *
     * <code>fmax(a,b) = b</code> if b's value is greater than or equal to a's.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is larger than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmax(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      return a.vi_->val_ > b.vi_->val_ ? a : b;
    }

    /**
     * Returns the maximum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a's value is greater than b,
     * then a is returned, otherwise a fesh variable implementation
     * wrapping the value b is returned.
     *
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is larger than or equal
     * to the second value, the first variable, otherwise the second
     * value promoted to a fresh variable.
     */
    inline var fmax(const stan::agrad::var& a,
                    const double& b) {
      return a.vi_->val_ >= b ? a : var(b);
    }

    /**
     * Returns the maximum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmax() for the double-based version.
     * 
     * For <code>fmax(a,b)</code>, if a is greater than b's value,
     * then a fresh variable implementation wrapping a is returned, otherwise 
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is larger than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmax(const double& a,
                    const stan::agrad::var& b) {
      return a > b.vi_->val_ ? var(a) : b;
    }

  }
}
#endif
