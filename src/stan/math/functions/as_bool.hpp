#ifndef STAN__MATH__FUNCTIONS__AS_BOOL_HPP
#define STAN__MATH__FUNCTIONS__AS_BOOL_HPP

namespace stan {
  namespace math {

    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param x Value.
     * @return 1 if argument is equal to zero and 0 otherwise.
     */
    template <typename T>
    inline int as_bool(const T x) {
      return x != 0.0;
    }

    /**
     * Return an integer with an equivalent boolean value to specified
     * input.  For integers, this reduces to the identity function.
     *
     * @param x value.
     * @return The value.
     */
    template <>
    inline int as_bool<int>(const int x) {
      return x;
    }


  }
}

#endif
