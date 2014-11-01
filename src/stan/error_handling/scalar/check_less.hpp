#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_LESS_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_LESS_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y, typename T_high, bool is_vec>
      struct less {
        static bool check(const char* function,
                          const char* name,  
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(high); n++) {
            if (!(y < high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than ";
              msg << high_vec[n];
              std::string message(msg.str());
              dom_err(function, name, y,
                      "is ", message.c_str());
            }
          }
          return true;
        }
      };
    
      template <typename T_y, typename T_high>
      struct less<T_y, T_high, true> {
        static bool check(const char* function,
                          const char* name,
                          const T_y& y,
                          const T_high& high) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y,n) < high_vec[n])) {
              std::stringstream msg;
              msg << ", but must be less than ";
              msg << high_vec[n];
              std::string message(msg.str());
              dom_err_vec(function, name, y, n,
                          "is ", message.c_str());
            }
          }
          return true;
        }
      };
    }

    // throws if any element of y or high is nan
    template <typename T_y, typename T_high>
    inline bool check_less(const char* function,
                           const char* name,  
                           const T_y& y,
                           const T_high& high) {
      return less<T_y, T_high, is_vector_like<T_y>::value>
        ::check(function, name, y, high);
    }
  }
}
#endif
