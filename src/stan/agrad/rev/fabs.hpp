#ifndef __STAN__AGRAD__REV__FABS_HPP__
#define __STAN__AGRAD__REV__FABS_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/precomp_v_vari.hpp>
#include <stan/agrad/rev/vari.hpp>
#include <stan/agrad/rev/operator_unary_negative.hpp>
#include <stan/math/constants.hpp>


namespace stan {
  namespace agrad {
    
    /**
     * Return the absolute value of the variable (cmath).  
     *
     * Choosing an arbitrary value at the non-differentiable point 0,
     * 
     * \f$\frac{d}{dx}|x| = \mbox{sgn}(x)\f$.
     *
     * where \f$\mbox{sgn}(x)\f$ is the signum function, taking values
     * -1 if \f$x < 0\f$, 0 if \f$x == 0\f$, and 1 if \f$x == 1\f$.
     *
     * The function <code>abs()</code> provides the same behavior, with
     * <code>abs()</code> defined in stdlib.h and <code>fabs()</code> defined in <code>cmath</code>.
     * The derivative is 0 if the input is 0.
     *
     * Returns std::numeric_limits<double>::quiet_NaN() for NaN inputs.
     *

     * @param a Input variable.
     * @return Absolute value of variable.
     */
    inline var fabs(const var& a) {
      using stan::math::NOT_A_NUMBER;
      // cut-and-paste from abs()
      if (a.val() > 0.0)
        return a;
      else if (a.val() < 0.0)
        return var(new neg_vari(a.vi_));
      else if (a.val() == 0)
        return var(new vari(0));
      else
        return var(new precomp_v_vari(NOT_A_NUMBER,a.vi_,NOT_A_NUMBER));
    }

  }
}
#endif
