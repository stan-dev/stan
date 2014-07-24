#ifndef STAN__MATH__ERROR_HANDLING_CHECK_NONNEGATIVE_HPP
#define STAN__MATH__ERROR_HANDLING_CHECK_NONNEGATIVE_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y, typename T_result, bool is_vec>
      struct nonnegative {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          // have to use not is_unsigned. is_signed will be false
          // floating point types that have no unsigned versions.
          if (!boost::is_unsigned<T_y>::value && !(y >= 0)) 
            return dom_err(function,y,name,
                           " is %1%, but must be >= 0!","",
                           result);
          return true;
        }
      };
    
      template <typename T_y, typename T_result>
      struct nonnegative<T_y, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const char* name,
                          T_result* result) {
          using stan::length;
          for (size_t n = 0; n < length(y); n++) {
            if (!boost::is_unsigned<typename T_y::value_type>::value 
                && !(stan::get(y,n) >= 0)) 
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be >= 0!","",
                                 result);
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_result>
    inline bool check_nonnegative(const char* function,
                                  const T_y& y,
                                  const char* name,
                                  T_result* result) {
      return nonnegative<T_y,T_result,is_vector_like<T_y>::value>
        ::check(function, y, name, result);
    }
  }
}
#endif
