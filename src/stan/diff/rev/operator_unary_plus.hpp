#ifndef __STAN__DIFF__REV__OPERATOR_UNARY_PLUS_HPP__
#define __STAN__DIFF__REV__OPERATOR_UNARY_PLUS_HPP__

#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {
    
    /**
     * Unary plus operator for variables (C++).  
     *
     * The function simply returns its input, because
     *
     * \f$\frac{d}{dx} +x = \frac{d}{dx} x = 1\f$.
     *
     * The effect of unary plus on a built-in C++ scalar type is
     * integer promotion.  Because variables are all 
     * double-precision floating point already, promotion is
     * not necessary.
     *
     * @param a Argument variable.
     * @return The input reference.
     */
    inline var operator+(const var& a) {
      return a;
    }

  }
}
#endif
