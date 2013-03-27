#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_FINITE_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_FINITE_HPP__

#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_result,
                class Policy,
                bool is_vec>
      struct finite {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          if (!(boost::math::isfinite)(y)) 
            return dom_err(function,y,name,
                           " is %1%, but must be finite!","",
                           result,Policy());
          return true;
        }
      };
    
      template <typename T_y,
                typename T_result,
                class Policy>
      struct finite<T_y, T_result, Policy, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!(boost::math::isfinite)(stan::get(y,n)))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be finite!","",
                                 result,Policy());
          }
          return true;
        }
      };
    }
    /**
     * Checks if the variable y is finite.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result,
                             const Policy&) {
      return finite<T_y,T_result,Policy,
                    is_vector_like<T_y>::value>
        ::check(function, y, name, result, Policy());
    }

    template <typename T_y, typename T_result>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result) {
      return check_finite(function,y,name,result,default_policy());
    }
    
    template <typename T>
    inline bool check_finite(const char* function,
                             const T& y,
                             const char* name) {
      return check_finite<T,typename scalar_type<T>::type *>
        (function,y,name,0,default_policy());
    }
 
  }
}
#endif
