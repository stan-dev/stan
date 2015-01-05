#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_POSITIVE_HPP

#include <boost/type_traits/is_unsigned.hpp>
#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/domain_error_vec.hpp>
#include <stan/math/meta/value_type.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace error_handling {

    namespace {

      template <typename T_y, bool is_vec>
      struct positive {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          // have to use not is_unsigned. is_signed will be false
          // floating point types that have no unsigned versions.
          if (!boost::is_unsigned<T_y>::value && !(y > 0)) 
            domain_error(function, name, y,
                    "is ", ", but must be > 0!");
          return true;
        }
      };
    
      template <typename T_y>
      struct positive<T_y, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          using stan::math::value_type;
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!boost::is_unsigned<typename value_type<T_y>::type>::value
                && !(stan::get(y,n) > 0)) 
              domain_error_vec(function, name, y, n,
                          "is ", ", but must be > 0!");
          }
          return true;
        }
      };

    }

    /**
     * Return <code>true</code> if <code>y</code> is positive.
     *
     * This function is vectorized and will check each element of
     * <code>y</code>. If any element is <code>NaN</code>, this
     * function will throw an exception.
     *
     * @tparam T_y Type of y
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Variable to check
     *
     * @return <code>true</code> if y is greater than 0.
     * @throw <code>domain_error</code> if y is negative or zero or
     *   if any element of y is NaN.
     */
    template <typename T_y>
    inline bool check_positive(const std::string& function,
                               const std::string& name,
                               const T_y& y) {
      return positive<T_y, is_vector_like<T_y>::value>
        ::check(function, name, y);
    }

  }
}
#endif
