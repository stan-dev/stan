#ifndef __STAN__AGRAD__REV__ABS_HPP__
#define __STAN__AGRAD__REV__ABS_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/operator_unary_negative.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the absolute value of the variable (std).  
     *
     * The value at the undifferentiable point 0 is conveniently set
     * 0, so that
     *
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * The function fabs() provides identical behavior, with
     * abs() defined in stdlib.h and fabs() defined in cmath.
     *
     * @param a Variable input.
     * @return Absolute value of variable.
     */
    inline var abs(const var& a) {   
      // cut-and-paste from fabs()
      if (a.val() > 0.0)
        return a;
      if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      // FIXME:  same as fabs() -- is this right?
      return var(new vari(0.0));
    }

  }
}
#endif
