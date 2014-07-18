#ifndef STAN__MATH__ERROR_HANDLING_CHECK_NOT_NAN_HPP
#define STAN__MATH__ERROR_HANDLING_CHECK_NOT_NAN_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, typename T_result, bool is_vec>
      struct not_nan {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          if ((boost::math::isnan)(y)) 
            return dom_err(function,y,name,
                           " is %1%, but must not be nan!","",
                           result);
          return true;
        }
      };
    
      template <typename T_y, typename T_result>
      struct not_nan<T_y, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          // using stan::length;
          for (size_t n = 0; n < stan::length(y); n++) {
            if ((boost::math::isnan)(stan::get(y,n)))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must not be nan!","",
                                 result);
          }
          return true;
        }
      };
    }

    /**
     * Checks if the variable y is nan.
     *
     * @param function Name of function being invoked.
     * @param y Reference to variable being tested.
     * @param name Name of variable being tested.
     * @param result Pointer to resulting value after test.
     * @tparam T_y Type of variable being tested.
     * @tparam T_result Type of result returned.
     */
    template <typename T_y, typename T_result>
    inline bool check_not_nan(const char* function,
                              const T_y& y,
                              const char* name,
                              T_result* result) {
      return not_nan<T_y,T_result,is_vector_like<T_y>::value>
        ::check(function, y, name, result);
    }

  }
}
#endif
