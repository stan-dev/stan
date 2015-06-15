#ifndef STAN_MATH_PRIM_SCAL_FUN_IDENTITY_FREE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_IDENTITY_FREE_HPP

namespace stan {

  namespace math {

    /**
     * Returns the result of applying the inverse of the identity
     * constraint transform to the input.
     *
     * <p>This method is effectively a no-op and mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param y Constrained scalar.
     * @return The input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T identity_free(const T y) {
      return y;
    }

  }

}

#endif
