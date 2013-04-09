#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_POSITIVE_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_POSITIVE_HPP__

#include <boost/type_traits/is_unsigned.hpp>
#include <stan/meta/traits.hpp>
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
      struct positive {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          // have to use not is_unsigned. is_signed will be false
          // floating point types that have no unsigned versions.
          if (!boost::is_unsigned<T_y>::value && !(y > 0)) 
            return dom_err(function,y,name,
                           " is %1%, but must be > 0!","",
                           result,Policy());
          return true;
        }
      };
    
      template <typename T_y,
                typename T_result,
                class Policy>
      struct positive<T_y, T_result, Policy, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!boost::is_unsigned<typename T_y::value_type>::value
                && !(stan::get(y,n) > 0)) 
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be > 0!","",
                                 result,Policy());
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const T_y& y,
                               const char* name,
                               T_result* result,
                               const Policy&) {
      return positive<T_y,T_result,Policy,
        is_vector_like<T_y>::value>
        ::check(function, y, name, result, Policy());
    }
    template <typename T_x, typename T_result>
    inline bool check_positive(const char* function,
                               const T_x& x,
                               const char* name,
                               T_result* result) {
      return check_positive(function,x,name,result,default_policy());
    }
    template <typename T>
    inline bool check_positive(const char* function,
                               const T& x,
                               const char* name) {
      return check_positive<T,typename scalar_type<T>::type *>
        (function,x,name,0,default_policy());
    }

  }
}
#endif
