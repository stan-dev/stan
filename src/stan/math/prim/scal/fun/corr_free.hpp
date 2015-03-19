#ifndef STAN__MATH__PRIM__SCAL__FUN__CORR_FREE_HPP
#define STAN__MATH__PRIM__SCAL__FUN__CORR_FREE_HPP

#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <cmath>

namespace stan {

  namespace prob {

    /**
     * Return the unconstrained scalar that when transformed to
     * a valid correlation produces the specified value.
     *
     * <p>This function inverts the transform defined for
     * <code>corr_constrain(T)</code>, which is the inverse hyperbolic
     * tangent,
     *
     * <p>\f$ f^{-1}(y)
     *          = \mbox{atanh}\, y
     *          = \frac{1}{2} \log \frac{y + 1}{y - 1}\f$.
     *
     * @param y Correlation scalar input.
     * @return Free scalar that transforms to the specified input.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    T corr_free(const T y) {
      stan::math::check_bounded<T,double,double>
        ("stan::prob::lub_free",
         "Correlation variable", y, -1, 1);
      return atanh(y);
    }

  }

}

#endif
