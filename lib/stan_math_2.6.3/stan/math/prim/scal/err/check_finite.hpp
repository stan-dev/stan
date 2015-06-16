#ifndef STAN_MATH_PRIM_SCAL_ERR_CHECK_FINITE_HPP
#define STAN_MATH_PRIM_SCAL_ERR_CHECK_FINITE_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/err/domain_error_vec.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/fun/value_of_rec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, bool is_vec>
      struct finite {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y) {
          using stan::math::value_of_rec;
          if (!(boost::math::isfinite)(value_of_rec(y)))
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
          using stan::math::value_of_rec;
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!(boost::math::isfinite)(value_of_rec(stan::get(y, n))))
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
