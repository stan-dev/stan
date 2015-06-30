#ifndef STAN_MATH_PRIM_SCAL_FUN_LOGICAL_NEGATION_HPP
#define STAN_MATH_PRIM_SCAL_FUN_LOGICAL_NEGATION_HPP

namespace stan {
  namespace math {

    /**
     * The logical negation function which returns 1 if the input
     * is equal to zero and 0 otherwise.
     *
     * @tparam T Type to compare to zero.
     * @param x Value to compare to zero.
     * @return 1 if input is equal to zero.
     */
    template <typename T>
    inline int
    logical_negation(const T x) {
      return (x == 0);
    }

  }
}

#endif
