#ifndef STAN__MATH__PRIM__SCAL__ERR__CHECK_NONNEGATIVE_HPP
#define STAN__MATH__PRIM__SCAL__ERR__CHECK_NONNEGATIVE_HPP

#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/err/domain_error_vec.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/arr/meta/length.hpp>
#include <stan/math/prim/scal/meta/value_type.hpp>
#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/arr/meta/is_vector_like.hpp>
#include <boost/type_traits/is_unsigned.hpp>

namespace stan {

  namespace math {

    namespace {
      template <typename T_y, bool is_vec>
      struct nonnegative {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y) {
          // have to use not is_unsigned. is_signed will be false
          // floating point types that have no unsigned versions.
          if (!boost::is_unsigned<T_y>::value && !(y >= 0))
            domain_error(function, name, y,
                         "is ", ", but must be >= 0!");
          return true;
        }
      };

      template <typename T_y>
      struct nonnegative<T_y, true> {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y) {
          using stan::length;
          using stan::math::value_type;

          for (size_t n = 0; n < length(y); n++) {
            if (!boost::is_unsigned<typename value_type<T_y>::type>::value
                && !(stan::get(y, n) >= 0))
              domain_error_vec(function, name, y, n,
                               "is ", ", but must be >= 0!");
          }
          return true;
        }
      };
    }

    /**
     * Return <code>true</code> if <code>y</code> is non-negative.
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
     * @return <code>true</code> if y is greater than or equal to 0.
     * @throw <code>domain_error</code> if y is negative or
     *   if any element of y is NaN.
     */
    template <typename T_y>
    inline bool check_nonnegative(const char* function,
                                  const char* name,
                                  const T_y& y) {
      return nonnegative<T_y, is_vector_like<T_y>::value>
        ::check(function, name, y);
    }
  }
}
#endif
