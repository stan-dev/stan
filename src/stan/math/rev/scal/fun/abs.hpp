#ifndef STAN__MATH__REV__SCAL__FUN__ABS_HPP
#define STAN__MATH__REV__SCAL__FUN__ABS_HPP

#include <stan/math/rev/scal/fun/fabs.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return the absolute value of the variable (std).
     *
     * Delegates to <code>fabs()</code> (see for doc).
     *
       \f[		
       \mbox{abs}(x) =
       \begin{cases}
         |x| & \mbox{if } -\infty\leq x\leq \infty \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]

       \f[
       \frac{\partial\,\mbox{abs}(x)}{\partial x} =
       \begin{cases}
         -1 & \mbox{if } x < 0 \\
         0 & \mbox{if } x = 0 \\
         1 & \mbox{if } x > 0 \\[6pt]
         \textrm{NaN} & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
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
