#ifndef STAN_MATH_PRIM_SCAL_FUN_STEP_HPP
#define STAN_MATH_PRIM_SCAL_FUN_STEP_HPP

namespace stan {
  namespace math {

    /**
     * The step, or Heaviside, function.
     *
     * The function is defined by
     *
     * <code>step(y) = (y < 0.0) ? 0 : 1</code>.
     *
       \f[
       \mbox{step}(x) =
       \begin{cases}
         0 & \mbox{if } x \leq 0 \\
         1 & \mbox{if } x > 0  \\[6pt]
         0 & \mbox{if } x = \textrm{NaN}
       \end{cases}
       \f]
     *
     * @param y Scalar argument.
     *
     * @return 1 if the specified argument is greater than or equal to
     * 0.0, and 0 otherwise.
     */
    template <typename T>
    inline int step(const T y) {
      return y < 0.0 ? 0 : 1;
    }

  }
}

#endif
