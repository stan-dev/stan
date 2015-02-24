#ifndef STAN__MATH__REV__CORE__STD_ISNAN_HPP
#define STAN__MATH__REV__CORE__STD_ISNAN_HPP

#include <stan/math/rev/core.hpp>
#include <cmath>

namespace std {

  /**
   * Checks if the given number is NaN.
   * 
   * Return <code>true</code> if the value of the
   * specified variable is not a number.
   *
   * @param a Variable to test.
   * @return <code>true</code> if value is not a number.
   */
  inline int isnan(const stan::agrad::var& a) {
    return isnan(a.val());
  }

}
#endif
