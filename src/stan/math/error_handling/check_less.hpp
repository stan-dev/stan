#ifndef STAN__MATH__ERROR_HANDLING_CHECK_LESS_HPP
#define STAN__MATH__ERROR_HANDLING_CHECK_LESS_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, typename T_high, typename T_result, bool is_vec>
      struct less {
        static bool check(const char* function,
                          const T_y& y,
                          const T_high& high,
                          const char* name,  
                          T_result* result) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(high); n++) {
            if (!(y < high_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be less than ",
                             high_vec[n],result);
          }
          return true;
        }
      };
    
      template <typename T_y, typename T_high, typename T_result>
      struct less<T_y, T_high, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_high& high,
                          const char* name,
                          T_result* result) {
          using stan::length;
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(stan::get(y,n) < high_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be less than ",
                                 high_vec[n],result);
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_high, typename T_result>
    inline bool check_less(const char* function,
                           const T_y& y,
                           const T_high& high,
                           const char* name,  
                           T_result* result) {
      return less<T_y,T_high,T_result,is_vector_like<T_y>::value>
        ::check(function,y,high,name,result);
    }
  }
}
#endif
