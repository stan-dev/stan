#ifndef STAN__MATH__ERROR_HANDLING__CHECK_BOUNDED_HPP
#define STAN__MATH__ERROR_HANDLING__CHECK_BOUNDED_HPP

#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/error_handling/dom_err_vec.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace math {

    namespace detail {
      
      // implemented using structs because there is no partial specialization
      // for templated functions
      
      // default implementation works for scalar T_y. T_low and T_high can
      // be either scalar or vector
      template <typename T_y, typename T_low, typename T_high, typename T_result,
                bool y_is_vec>
      struct bounded {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const T_high& high,
                          const char* name,  
                          T_result* result) {
          using stan::length;
          using stan::max_size;
          typedef std::pair<typename scalar_type<T_low>::type, 
                            typename scalar_type<T_high>::type> pair_type;

          VectorView<const T_low> low_vec(low);
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < max_size(low, high); n++) {
            if (!(low_vec[n] <= y && y <= high_vec[n]))
              return dom_err(function,y,name,
                             " is %1%, but must be between ",
                             pair_type(low_vec[n], high_vec[n]),
                             result);
          }
          return true;
        }
      };
    
      template <typename T_y,
                typename T_low,
                typename T_high,
                typename T_result>
      struct bounded<T_y, T_low, T_high, T_result, true> {
        static bool check(const char* function,
                          const T_y& y,
                          const T_low& low,
                          const T_high& high,
                          const char* name,
                          T_result* result) {
          using stan::length;
          using stan::get;
          typedef std::pair<typename scalar_type<T_low>::type, 
                            typename scalar_type<T_high>::type> pair_type;
          
          VectorView<const T_low> low_vec(low);
          VectorView<const T_high> high_vec(high);
          for (size_t n = 0; n < length(y); n++) {
            if (!(low_vec[n] <= get(y,n) && get(y,n) <= high_vec[n]))
              return dom_err_vec(n,function,y,name,
                                 " is %1%, but must be between ",
                                 pair_type(low_vec[n], high_vec[n]),
                                 result);
          }
          return true;
        }
      };
    }

    // public check_bounded function
    template <typename T_y, typename T_low, typename T_high, typename T_result>
    inline bool check_bounded(const char* function,
                              const T_y& y,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result) {
      return detail::bounded<T_y,T_low,T_high,T_result,
                             is_vector_like<T_y>::value>
        ::check(function,y,low,high,name,result);
    }

  }
}
#endif
