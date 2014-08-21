#ifndef STAN__AGRAD__REV__FUNCTIONS__ABS_HPP
#define STAN__AGRAD__REV__FUNCTIONS__ABS_HPP

#include <stan/agrad/rev/functions/fabs.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the absolute value of the variable (std).  
     *
     * Delegates to <code>fabs()</code> (see for doc).
     * 
     * @param a Variable input.
     * @return Absolute value of variable.
     */
    inline var abs(const var& a) { 
      return fabs(a);
    }

  }
}
#endif
