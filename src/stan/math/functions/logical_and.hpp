#ifndef STAN__MATH__FUNCTIONS__LOGICAL_AND_HPP
#define STAN__MATH__FUNCTIONS__LOGICAL_AND_HPP

namespace stan {
  namespace math {

    /**
     * The logical and function which returns 1 if both arguments
     * are unequal to zero and 0 otherwise. 
     * Equivalent
     * to <code>x1 != 0 && x2 != 0</code>.
     *
     * @tparam T1 Type of first argument.
     * @tparam T2 Type of second argument.
     * @param x1 First argument
     * @param x2 Second argument
     * @return <code>true</code> if both x1 and x2 are not equal to 0.
     */
    template <typename T1, typename T2>
    inline int
    logical_and(const T1 x1, const T2 x2) {
      return (x1 != 0) && (x2 != 0);
    }

  }
}

#endif
