#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_FINITE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_FINITE_HPP

#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/domain_error_vec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, bool is_vec>
      struct finite {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y) {
          if (!(boost::math::isfinite)(y))
            domain_error(function, name, y,
                         "is ", ", but must be finite!");
          return true;
        }
      };
    
      template <typename T_y>
      struct finite<T_y, true> {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!(boost::math::isfinite)(stan::get(y,n)))
              domain_error_vec(function, name, y, n,
                               "is ", ", but must be finite!");
          }
          return true;
        }
      };
    }
    
    /**
     * Return <code>true</code> if <code>y</code> is finite.
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
     * @return <code>true</code> if y is finite.
     * @throw <code>domain_error</code> if y is infinity, -infinity, or
     *   NaN.
     */
    template <typename T_y>
    inline bool check_finite(const char* function,
                             const char* name,
                             const T_y& y) {
      return finite<T_y, is_vector_like<T_y>::value>
        ::check(function, name, y);
    }
  }
}
#endif
