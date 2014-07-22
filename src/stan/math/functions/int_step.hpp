#ifndef STAN__MATH__FUNCTIONS__INT_STEP_HPP
#define STAN__MATH__FUNCTIONS__INT_STEP_HPP

namespace stan {
  namespace math {
    /**
     * The integer step, or Heaviside, function.  
     *
     * @param y Value to test.
     * @return 1 if value is greater than 0 and 0 otherwise
     * @tparam T Scalar argument type.
     */
    template <typename T>
    unsigned int int_step(const T y) {
      return y > 0;
    }
  }
}

#endif
