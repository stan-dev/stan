#ifndef STAN__MATH__ERROR_HANDLING_CHECK_FINITE_HPP
#define STAN__MATH__ERROR_HANDLING_CHECK_FINITE_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_result,
                bool is_vec>
      struct finite {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          if (!(boost::math::isfinite)(y)) 
            return dom_err(function,y,name,
                           " is %1%, but must be finite!","",
                           result);
          return true;
        }
      };
    
      template <typename T_y, typename T_result>
      struct finite<T_y, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!(boost::math::isfinite)(stan::get(y,n)))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be finite!","",
                                 result);
          }
          return true;
        }
      };
    }
    
    /**
     * Checks if the variable y is finite.
     */
    template <typename T_y, typename T_result>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result) {
      return finite<T_y,T_result,is_vector_like<T_y>::value>
        ::check(function, y, name, result);
    }
  }
}
#endif
