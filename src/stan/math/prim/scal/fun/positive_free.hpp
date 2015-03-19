#ifndef STAN__MATH__PRIM__SCAL__FUN__POSITIVE_FREE_HPP
#define STAN__MATH__PRIM__SCAL__FUN__POSITIVE_FREE_HPP

#include <stan/math/prim/scal/err/check_positive.hpp>
#include <cmath>

namespace stan {

  namespace prob {

    /**
     * Return the unconstrained value corresponding to the specified
     * positive-constrained value.
     *
     * <p>The transform is the inverse of the transform \f$f\f$ applied by
     * <code>positive_constrain(T)</code>, namely
     *
     * <p>\f$f^{-1}(x) = \log(x)\f$.
     *
     * <p>The input is validated using <code>stan::math::check_positive()</code>.
     *
     * @param y Input scalar.
     * @return Unconstrained value that produces the input when constrained.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the variable is negative.
     */
    template <typename T>
    inline
    T positive_free(const T y) {
      stan::math::check_positive("stan::prob::positive_free", "Positive variable", y);
      return log(y);
    }

  }

}

#endif
