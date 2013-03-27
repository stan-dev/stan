#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_GREATER_OR_EQUAL_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_GREATER_OR_EQUAL_HPP__

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_low,
                typename T_result,
                class Policy,
                bool is_vec>
      struct greater_or_equal {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const char* name,  
                          T_result* result,
                          const Policy&) {
          using stan::length;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(low); n++) {
            if (!(y >= low_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be greater than ",
                             low_vec[n],result,Policy());
          }
          return true;
        }
      };
      
      template <typename T_y,
                typename T_low,
                typename T_result,
                class Policy>
      struct greater_or_equal<T_y, T_low, T_result, Policy, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          using stan::length;
          using stan::get;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(y); n++) {
            if (!(get(y,n) >= low_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be greater than or equal to ",
                                 low_vec[n],result,Policy());
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_low, typename T_result, class Policy>
    inline bool check_greater_or_equal(const char* function,
                                       const T_y& y,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result,
                                       const Policy&) {
      return greater_or_equal<T_y,T_low,T_result,Policy,is_vector_like<T_y>::value>::check(function,y,low,name,result,Policy());
    }
    template <typename T_y, typename T_low, typename T_result>
    inline bool check_greater_or_equal(const char* function,
                                       const T_y& y,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result) {
      return check_greater_or_equal(function,y,low,name,result,
                                    default_policy());
    }                               
    template <typename T_y, typename T_low>
    inline bool check_greater_or_equal(const char* function,
                                       const T_y& y,
                                       const T_low& low,
                                       const char* name) {
      return check_greater_or_equal<T_y,T_low,typename scalar_type<T_y>::type *>
        (function,y,low,name,0,default_policy());
    }

  }
}
#endif
