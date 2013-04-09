#ifndef __STAN__MATH__ERROR_HANDLING__CHECK_BOUNDED_HPP__
#define __STAN__MATH__ERROR_HANDLING__CHECK_BOUNDED_HPP__

#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>

namespace stan {
  namespace math {

    namespace {
      template <typename T_y,
                typename T_low,
                typename T_high,
                typename T_result,
                class Policy,
                bool is_vec>
      struct bounded {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const T_high& high,
                          const char* name,  
                          T_result* result,
                          const Policy&) {
          using stan::length;
          using stan::max_size;
          VectorView<const T_low> low_vec(low);
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < max_size(low, high); n++) {
            if (!(low_vec[n] <= y && y <= high_vec[n]))
              return dom_err(function,y,name," is %1%, but must be between ",
                             std::pair<typename scalar_type<T_low>::type, 
                                       typename scalar_type<T_high>::type>(low_vec[n],
                                                                           high_vec[n]),
                             result,Policy());
          }
          return true;
        }
      };
    
      template <typename T_y,
                typename T_low,
                typename T_high,
                typename T_result,
                class Policy>
      struct bounded<T_y, T_low, T_high, T_result, Policy, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const T_high& high,
                          const char* name,
                          T_result* result,
                          const Policy&) {
          using stan::length;
          using stan::get;
          VectorView<const T_low> low_vec(low);
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(low_vec[n] <= get(y,n) && get(y,n) <= high_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be between ",
                                 std::pair<typename scalar_type<T_low>::type, 
                                           typename scalar_type<T_high>::type>(low_vec[n],
                                                                               high_vec[n]),
                                 result,Policy());
          }
          return true;
        }
      };
    }
    template <typename T_y, typename T_low, typename T_high, typename T_result, class Policy>
    inline bool check_bounded(const char* function,
                              const T_y& y,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result,
                              const Policy&) {
      return bounded<T_y,T_low,T_high,T_result,Policy,
                     is_vector_like<T_y>::value>
        ::check(function,y,low,high,name,result,Policy());
    }
    template <typename T_y, typename T_low, typename T_high, typename T_result>
    inline bool check_bounded(const char* function,
                              const T_y& y,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result) {
      return check_bounded(function,y,low,high,name,result,default_policy());
    }
    template <typename T_y, typename T_low, typename T_high>
    inline bool check_bounded(const char* function,
                              const T_y& y,
                              const T_low& low,
                              const T_high& high,
                              const char* name) {
      return check_bounded<T_y,T_low,T_high,typename scalar_type<T_y>::type *>
        (function,y,low,high,name,0,default_policy());
    }

  }
}
#endif
