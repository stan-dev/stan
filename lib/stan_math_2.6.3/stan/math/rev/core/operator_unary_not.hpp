#ifndef STAN_MATH_REV_CORE_OPERATOR_UNARY_NOT_HPP
#define STAN_MATH_REV_CORE_OPERATOR_UNARY_NOT_HPP

#include <stan/math/rev/core/var.hpp>

namespace stan {
  namespace math {

    /**
     * Prefix logical negation for the value of variables (C++).  The
     * expression (!a) is equivalent to negating the scalar value of
     * the variable a.
     *
     * Note that this is the only logical operator defined for
     * variables.  Overridden logical conjunction (&&) and disjunction
     * (||) operators do not apply the same "short circuit" rules
     * as the built-in logical operators.
     *
        \f[
        \mbox{operator!}(x) =
        \begin{cases}
          0 & \mbox{if } x \neq 0 \\
          1 & \mbox{if } x = 0 \\[6pt]
          0 & \mbox{if } x = \textrm{NaN}
        \end{cases}
        \f]
     *
     * @param a Variable to negate.
     * @return True if variable is non-zero.
     */
    inline bool operator!(const var& a) {
      return !a.val();
    }

  }
}
#endif
