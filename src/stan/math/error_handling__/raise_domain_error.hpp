#ifndef __STAN__MATH__ERROR_HANDLING__RAISE_DOMAIN_ERROR_HPP__
#define __STAN__MATH__ERROR_HANDLING__RAISE_DOMAIN_ERROR_HPP__

#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

namespace stan {
  namespace math {
    namespace policies {
      namespace detail {
        using boost::math::policies::detail::raise_error;
        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* function, 
                                           const char* message, 
                                           const T_val& val, 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::throw_on_error>&) {
          raise_error<std::domain_error, T_val>(function, message, val);
          // we never get here:
          return std::numeric_limits<T_result>::quiet_NaN();
        }

        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* , 
                                           const char* , 
                                           const T_val& , 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::ignore_error>&) {
          // This may or may not do the right thing, but the user asked for the error
          // to be ignored so here we go anyway:
          return std::numeric_limits<T_result>::quiet_NaN();
        }

        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* , 
                                           const char* , 
                                           const T_val& , 
                                           const ::boost::math::policies::domain_error< ::boost::math::policies::errno_on_error>&) {
          errno = EDOM;
          // This may or may not do the right thing, but the user asked for the error
          // to be silent so here we go anyway:
          return std::numeric_limits<T_result>::quiet_NaN();
        }
       
        template <class T_result, class T_val>
        inline T_result raise_domain_error(const char* function, 
                                           const char* message, 
                                           const T_val& val, 
                                           const  ::boost::math::policies::domain_error< ::boost::math::policies::user_error>&) {
          return user_domain_error(function, message, val);
        }

      }
    
      template <class T_result, class T_val, class Policy>
      inline T_result raise_domain_error(const char* function, 
                          const char* message, 
                          const T_val& val, 
                          const Policy&) {
        typedef typename Policy::domain_error_type policy_type;
        return detail::raise_domain_error<T_result,T_val>(function, 
                                                          message ? message : "Domain Error evaluating function at %1%", 
                                                          val, 
                                                          policy_type());
      
      }
    }
  }
  
}


#endif
