#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_LESS_OR_EQUAL_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_LESS_OR_EQUAL_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y, typename T_high, bool is_vec>
      struct less_or_equal {
        static bool check(const std::string& function,
                          const std::string& name,  
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(high); n++) {
            if (!(y <= high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than or equal to ";
              msg << high_vec[n];
              dom_err(function, name, y,
                      "is ", msg.str());
            }
          }
          return true;
        }
      };
    
      template <typename T_y, typename T_high>
      struct less_or_equal<T_y, T_high, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y,n) <= high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than or equal to ";
              msg << high_vec[n];
              dom_err_vec(function, name, y, n,
                          "is ", msg.str());
            }
          }
          return true;
        }
      };
    }
    
    // throws if any element of y or high is nan
    template <typename T_y, typename T_high>
    inline bool check_less_or_equal(const std::string& function,
                                    const std::string& name,  
                                    const T_y& y,
                                    const T_high& high) {
      return less_or_equal<T_y, T_high, is_vector_like<T_y>::value>
        ::check(function, name, y, high);
    }

  }
}
#endif
