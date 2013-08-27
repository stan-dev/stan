#ifndef __STAN__DIFF__REV__OPERATOR_NOT_EQUAL_HPP__
#define __STAN__DIFF__REV__OPERATOR_NOT_EQUAL_HPP__

#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {

    /**
     * Inequality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is not the same as the
     * second's.
     */
    inline bool operator!=(const var& a, const var& b) {
      return a.val() != b.val();
    }

    /**
     * Inequality operator comparing a variable's value and a double
     * (C++).
     *
     * @param a First variable.  
     * @param b Second value.
     * @return True if the first variable's value is not the same as the
     * second value.
     */
    inline bool operator!=(const var& a, const double b) {
      return a.val() != b;
    }

    /**
     * Inequality operator comparing a double and a variable's value
     * (C++).
     *
     * @param a First value.
     * @param b Second variable. 
     * @return True if the first value is not the same as the
     * second variable's value.
     */
    inline bool operator!=(const double a, const var& b) {
      return a != b.val();
    }

  }
}
#endif
