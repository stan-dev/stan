#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_EQUAL_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_EQUAL_HPP

#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/scalar/dom_err_vec.hpp>

namespace stan {
  namespace error_handling {

    namespace {
      template <typename T_y,
                typename T_eq,
                bool is_vec>
      struct equal {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y,
                          const T_eq& eq) {
          using stan::length;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(eq); n++) {
            if (!(y == eq_vec[n])) {
              std::stringstream msg;
              msg << ", but must be equal to ";
              msg << eq_vec[n];
              dom_err(function, name, y,
                      "is ", msg.str());
            }
          }
          return true;
        }
      };
      
      // throws if y or eq is nan
      template <typename T_y,
                typename T_eq>
      struct equal<T_y, T_eq, true> {
        static bool check(const std::string& function,
                          const std::string& name,
                          const T_y& y,
                          const T_eq& eq) {
          using stan::length;
          using stan::get;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(y); n++) {
            if (!(get(y,n) == eq_vec[n])) {
              std::stringstream msg;
              msg << ", but must be equal to ";
              msg << eq_vec[n];
              dom_err_vec(function, name, y, n,
                          "is ", msg.str());
            }
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_eq>
    inline bool check_equal(const std::string& function,
                            const std::string& name,
                            const T_y& y,
                            const T_eq& eq) {
      return equal<T_y, T_eq, is_vector_like<T_y>::value>
        ::check(function, name, y, eq);
    }
  }
}
#endif
