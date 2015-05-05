#ifndef STAN_MATH_PRIM_SCAL_FUN_AS_BOOL_HPP
#define STAN_MATH_PRIM_SCAL_FUN_AS_BOOL_HPP

namespace stan {
  namespace math {

    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param x Value.
     * @return 1 if argument is equal to zero (or NaN) and 0 otherwise.
     */
    template <typename T>
    inline bool as_bool(const T x) {
      return x != 0;
    }

  }
}

#endif
