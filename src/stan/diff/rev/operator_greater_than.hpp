#ifndef __STAN__DIFF__REV__OPERATOR_GREATER_THAN_HPP__
#define __STAN__DIFF__REV__OPERATOR_GREATER_THAN_HPP__

#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {

    /**
     * Greater than operator comparing variables' values (C++).
     *
     * @param a First variable.
     * @param b Second variable.
     * @return True if first variable's value is greater than second's.
     */
    inline bool operator>(const var& a, const var& b) {
      return a.val() > b.val();
    }

    /**
     * Greater than operator comparing variable's value and double
     * (C++).
     *
     * @param a First variable.
     * @param b Second value.
     * @return True if first variable's value is greater than second value.
     */
    inline bool operator>(const var& a, const double b) {
      return a.val() > b;
    }

    /**
     * Greater than operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable.
     * @return True if first value is greater than second variable's value.
     */
    inline bool operator>(const double a, const var& b) {
      return a > b.val();
    }

  }
}
#endif
