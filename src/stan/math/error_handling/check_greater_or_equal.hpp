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
                bool is_vec>
      struct greater_or_equal {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const char* name,  
                          T_result* result) {
          using stan::length;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(low); n++) {
            if (!(y >= low_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be greater than ",
                             low_vec[n],result);
          }
          return true;
        }
      };
      
      template <typename T_y,
                typename T_low,
                typename T_result>
      struct greater_or_equal<T_y, T_low, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const char* name,
                          T_result* result) {
          using stan::length;
          using stan::get;
          VectorView<const T_low> low_vec(low);
          for (size_t n = 0; n < length(y); n++) {
            if (!(get(y,n) >= low_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be greater than or equal to ",
                                 low_vec[n],result);
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_low, typename T_result>
    inline bool check_greater_or_equal(const char* function,
                                       const T_y& y,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result) {
      return greater_or_equal<T_y,T_low,T_result,is_vector_like<T_y>::value>
        ::check(function,y,low,name,result);
    }
   template <typename T_y, typename T_low>
    inline bool check_greater_or_equal(const char* function,
                                       const T_y& y,
                                       const T_low& low,
                                       const char* name) {
      return check_greater_or_equal<T_y,T_low,typename scalar_type<T_y>::type *>
        (function,y,low,name,0);
    }

  }
}
#endif
