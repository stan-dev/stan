#ifndef __STAN__AGRAD__REV__ABS_HPP__
#define __STAN__AGRAD__REV__ABS_HPP__

#include <stan/agrad/rev/fabs.hpp>

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
