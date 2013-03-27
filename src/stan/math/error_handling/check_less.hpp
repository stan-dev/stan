#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_LESS_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_LESS_HPP__

#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_high,
                typename T_result,
                class Policy,
                bool is_vec>
      struct less {
        static bool check(const char* function,
                          const T_y& y,
                          const T_high& high,
                          const char* name,  
                          T_result* result,
                          const Policy&) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(high); n++) {
            if (!(y < high_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be less than ",
                             high_vec[n],result,Policy());
          }
          return true;
        }
      };
    
      template <typename T_y,
                typename T_high,
                typename T_result,
                class Policy>
      struct less<T_y, T_high, T_result, Policy, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_high& high,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y,n) < high_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be less than ",
                                 high_vec[n],result,Policy());
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_high, typename T_result, class Policy>
    inline bool check_less(const char* function,
                           const T_y& y,
                           const T_high& high,
                           const char* name,  
                           T_result* result,
                           const Policy&) {
      return less<T_y,T_high,T_result,Policy,is_vector_like<T_y>::value>
        ::check(function,y,high,name,result,Policy());
    }
    template <typename T_y, typename T_high, typename T_result>
    inline bool check_less(const char* function,
                           const T_y& y,
                           const T_high& high,
                           const char* name,  
                           T_result* result) {
      return check_less(function,y,high,name,result,default_policy());
    }
    template <typename T_y, typename T_high>
    inline bool check_less(const char* function,
                           const T_y& y,
                           const T_high& high,
                           const char* name) {
      return check_less<T_y,T_high,typename scalar_type<T_y>::type *>
        (function,y,high,name,0,default_policy());
    }

  }
}
#endif
