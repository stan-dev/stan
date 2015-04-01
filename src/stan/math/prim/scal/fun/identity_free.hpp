#ifndef STAN__MATH__PRIM__SCAL__FUN__IDENTITY_FREE_HPP
#define STAN__MATH__PRIM__SCAL__FUN__IDENTITY_FREE_HPP

namespace stan {

  namespace prob {

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
