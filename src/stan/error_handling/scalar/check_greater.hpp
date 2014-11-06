#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_GREATER_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_GREATER_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y,
                typename T_low,
                bool is_vec>
      struct greater {
        static bool check(const std::string& function,
                          const std::string& name,  
                          const T_y& y,
                          const T_low& low) {
          using stan::length;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(low); n++) {
            if (!(y > low_vec[n])) {
              std::stringstream msg;
              msg << ", but must be greater than ";
              msg << low_vec[n];

              dom_err(function, name, y,
                      "is ", msg.str());
            }
          }
          return true;
        }
      };
    
      template <typename T_y,
                typename T_low>
      struct greater<T_y, T_low, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y,
                          const T_low& low) {
          using stan::length;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y,n) > low_vec[n])) {
              std::stringstream msg;
              msg << ", but must be greater than ";
              msg << low_vec[n];
              dom_err_vec(function, name, y, n,
                          "is ", msg.str());
            }
          }
          return true;
        }
      };
    }
    
    // throws if any element in y or low is nan
    template <typename T_y, typename T_low>
    inline bool check_greater(const std::string& function,
                              const std::string& name,  
                              const T_y& y,
                              const T_low& low) {
      return greater<T_y, T_low, is_vector_like<T_y>::value>
        ::check(function, name, y, low);
    }
 
  }
}
#endif
