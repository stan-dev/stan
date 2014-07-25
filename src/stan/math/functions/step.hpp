#ifndef STAN__MATH__FUNCTIONS__STEP_HPP
#define STAN__MATH__FUNCTIONS__STEP_HPP

namespace stan {
  namespace math {

    /**
     * The step, or Heaviside, function.  
     *
     * The function is defined by 
     *
     * <code>step(y) = (y < 0.0) ? 0 : 1</code>.
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
