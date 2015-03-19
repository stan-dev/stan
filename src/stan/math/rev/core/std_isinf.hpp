#ifndef STAN__MATH__REV__CORE__STD_ISINF_HPP
#define STAN__MATH__REV__CORE__STD_ISINF_HPP

#include <stan/math/rev/core/var.hpp>
#include <cmath>

namespace std {

  /**
   * Checks if the given number is infinite.
   *
   * Return <code>true</code> if the value of the
   * a is positive or negative infinity.
   *
   * @param a Variable to test.
   * @return <code>true</code> if value is infinite.
   */
  inline int isinf(const stan::agrad::var& a) {
    return isinf(a.val());
  }

}
#endif
