#ifndef __STAN__AGRAD__REV__OPERATOR_EQUALITY_HPP__
#define __STAN__AGRAD__REV__OPERATOR_EQUALITY_HPP__

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Equality operator comparing two variables' values (C++).
     *
     * @param a First variable.  
     * @param b Second variable. 
     * @return True if the first variable's value is the same as the
     * second's.
     */
    inline bool operator==(const var& a, const var& b) {
      return a.val() == b.val();
    }

  }
}
#endif
