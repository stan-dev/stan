#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_FINITE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_FINITE_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y, bool is_vec>
      struct finite {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          if (!(boost::math::isfinite)(y))
            dom_err(function, name, y,
                    "is ", ", but must be finite!");
          return true;
        }
      };
    
      template <typename T_y>
      struct finite<T_y, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!(boost::math::isfinite)(stan::get(y,n)))
              dom_err_vec(function, name, y, n,
                          "is ", ", but must be finite!");
          }
          return true;
        }
      };
    }
    
    /**
     * Checks if the variable y is finite.
     * NOTE: throws if any element in y is nan.
     */
    template <typename T_y>
    inline bool check_finite(const std::string& function,
                             const std::string& name,
                             const T_y& y) {
      return finite<T_y, is_vector_like<T_y>::value>
        ::check(function, name, y);
    }
  }
}
#endif
