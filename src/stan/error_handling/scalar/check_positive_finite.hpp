#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_FINITE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_FINITE_HPP

#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>

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
    inline bool check_positive_finite(const std::string& function,
                                      const std::string& name,
                                      const T_y& y) {
      stan::math::check_positive(function, name, y);
      stan::math::check_finite(function, name, y);

      return true;
    }

  }
}
#endif
