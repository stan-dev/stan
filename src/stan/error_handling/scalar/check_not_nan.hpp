#ifndef STAN__ERROR_HANDLING_CHECK_NOT_NAN_HPP
#define STAN__ERROR_HANDLING_CHECK_NOT_NAN_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y, bool is_vec>
      struct not_nan {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          if ((boost::math::isnan)(y)) 
            dom_err(function, name, y,
                    "is ", ", but must not be nan!");
          return true;
        }
      };
    
      template <typename T_y>
      struct not_nan<T_y, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y) {
          // using stan::length;
          for (size_t n = 0; n < stan::length(y); n++) {
            if ((boost::math::isnan)(stan::get(y,n)))
              dom_err_vec(function, name, y, n,
                          "is ", ", but must not be nan!");
          }
          return true;
        }
      };
    }

    /**
     * Checks if the variable y is nan.
     *
     * @param function Name of function being invoked.
     * @param name Name of variable being tested.
     * @param y Reference to variable being tested.
     * @tparam T_y Type of variable being tested.
     */
    template <typename T_y>
    inline bool check_not_nan(const std::string& function,
                              const std::string& name,
                              const T_y& y) {
      return not_nan<T_y, is_vector_like<T_y>::value>
        ::check(function, name, y);
    }

  }
}
#endif
