#ifndef STAN__AGRAD__REV__FUNCTIONS__FMIN_HPP
#define STAN__AGRAD__REV__FUNCTIONS__FMIN_HPP

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Returns the minimum of the two variable arguments (C99).
     *
     * See boost::math::fmin() for the double-based version.
     *
     * For <code>fmin(a,b)</code>, if a's value is less than b's,
     * then a is returned, otherwise b is returned.
     * 
     * @param a First variable.
     * @param b Second variable.
     * @return If the first variable's value is smaller than the
     * second's, the first variable, otherwise the second variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const stan::agrad::var& b) {
      return a.vi_->val_ < b.vi_->val_ ? a : b;
    }

    /**
     * Returns the minimum of the variable and scalar, promoting the
     * scalar to a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a's value is less than b, then a
     * is returned, otherwise a fresh variable wrapping b is returned.
     * 
     * @param a First variable.
     * @param b Second value
     * @return If the first variable's value is less than or equal to the second value,
     * the first variable, otherwise the second value promoted to a fresh variable.
     */
    inline var fmin(const stan::agrad::var& a,
                    const double& b) {
      return a.vi_->val_ <= b ? a : var(b);
    }

    /**
     * Returns the minimum of a scalar and variable, promoting the scalar to
     * a variable if it is larger (C99).
     *
     * See boost::math::fmin() for the double-based version.
     * 
     * For <code>fmin(a,b)</code>, if a is less than b's value, then a
     * fresh variable implementation wrapping a is returned, otherwise
     * b is returned.
     *
     * @param a First value.
     * @param b Second variable.
     * @return If the first value is smaller than the second variable's value,
     * return the first value promoted to a variable, otherwise return the 
     * second variable.
     */
    inline var fmin(const double& a,
                    const stan::agrad::var& b) {
      return a < b.vi_->val_ ? var(a) : b;
    }

  }
}
#endif
