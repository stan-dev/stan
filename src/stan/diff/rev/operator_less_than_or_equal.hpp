#ifndef __STAN__DIFF__REV__OPERATOR_LESS_THAN_OR_EQUAL_HPP__
#define __STAN__DIFF__REV__OPERATOR_LESS_THAN_OR_EQUAL_HPP__

#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {

    /**
     * Less than or equal operator comparing two variables' values
     * (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is less than or equal to
     * the second's.
     */
    inline bool operator<=(const var& a, const var& b) {
      return a.val() <= b.val();
    }

    /**
     * Less than or equal operator comparing a variable's value and a
     * scalar (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is less than or equal to
     * the second value.
     */
    inline bool operator<=(const var& a, const double b) {
      return a.val() <= b;
    }

    /**
     * Less than or equal operator comparing a double and variable's
     * value (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is less than or equal to the second
     * variable's value.
     */
    inline bool operator<=(const double a, const var& b) {
      return a <= b.val();
    }

  }
}
#endif
