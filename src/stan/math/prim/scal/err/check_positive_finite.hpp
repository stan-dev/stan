#ifndef STAN_MATH_PRIM_SCAL_ERR_CHECK_POSITIVE_FINITE_HPP
#define STAN_MATH_PRIM_SCAL_ERR_CHECK_POSITIVE_FINITE_HPP

#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if <code>y</code> is positive and finite.
     *
     * This function is vectorized and will check each element of
     * <code>y</code>.
     *
     * @tparam T_y Type of y
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Variable to check
     *
     * @return <code>true</code> if every element of y is greater than 0
     *   and y is not infinite.
     * @throw <code>domain_error</code> if any element of y is not positive or
     *   if any element of y is NaN.
     */
    template <typename T_y>
    inline bool check_positive_finite(const char* function,
                                      const char* name,
                                      const T_y& y) {
      stan::math::check_positive(function, name, y);
      stan::math::check_finite(function, name, y);

      return true;
    }

  }
}
#endif
