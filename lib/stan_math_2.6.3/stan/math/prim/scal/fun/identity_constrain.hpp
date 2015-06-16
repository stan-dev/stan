#ifndef STAN_MATH_PRIM_SCAL_FUN_IDENTITY_CONSTRAIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_IDENTITY_CONSTRAIN_HPP

namespace stan {

  namespace math {

    /**
     * Returns the result of applying the identity constraint
     * transform to the input.
     *
     * <p>This method is effectively a no-op and is mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * @return Transformed input.
     *
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T identity_constrain(T x) {
      return x;
    }

    /**
     * Returns the result of applying the identity constraint
     * transform to the input and increments the log probability
     * reference with the log absolute Jacobian determinant.
     *
     * <p>This method is effectively a no-op and mainly useful as a
     * placeholder in auto-generated code.
     *
     * @param x Free scalar.
     * lp Reference to log probability.
     * @return Transformed input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T identity_constrain(const T x, T& /*lp*/) {
      return x;
    }

  }

}

#endif
