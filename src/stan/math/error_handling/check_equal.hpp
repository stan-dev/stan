#ifndef STAN__MATH__ERROR_HANDLING_CHECK_EQUAL_HPP
#define STAN__MATH__ERROR_HANDLING_CHECK_EQUAL_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_eq,
                typename T_result,
                bool is_vec>
      struct equal {
        static bool check(const char* function,
                          const T_y& y,
                          const T_eq& eq,
                          const char* name,  
                          T_result* result) {
          using stan::length;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(eq); n++) {
            if (!(y == eq_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be equal to ",
                             eq_vec[n],result);
          }
          return true;
        }
      };
      
      template <typename T_y,
                typename T_eq,
                typename T_result>
      struct equal<T_y, T_eq, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_eq& eq,
                          const char* name,
                          T_result* result) {
          using stan::length;
          using stan::get;
          VectorView<const T_eq> eq_vec(eq);
          for (size_t n = 0; n < length(y); n++) {
            if (!(get(y,n) == eq_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be equal to ",
                                 eq_vec[n],result);
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_eq, typename T_result>
    inline bool check_equal(const char* function,
                            const T_y& y,
                            const T_eq& eq,
                            const char* name,  
                            T_result* result) {
      return equal<T_y,T_eq,T_result,is_vector_like<T_y>::value>
        ::check(function,y,eq,name,result);
    }
  }
}
#endif
