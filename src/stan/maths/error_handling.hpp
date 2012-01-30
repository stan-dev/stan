#ifndef __STAN__MATHS__ERROR_HANDLING_HPP__
#define __STAN__MATHS__ERROR_HANDLING_HPP__

#include <limits>

#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

#include <stan/maths/special_functions.hpp>

namespace stan { 

  namespace maths {
    const double CONSTRAINT_TOLERANCE = 1E-8;

    /**
     * Default error-handling policy from Boost.
     */
    typedef boost::math::policies::policy<> default_policy;

    /**
     * Checks if the variable y is nan.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const T_y& y,
                              const char* name,
                              T_result* result,
                              const Policy& /* pol */) {
      using boost::math::policies::raise_domain_error;
      if (boost::math::isnan(y)) {
        std::string msg_str(name);
        msg_str += " is %1%, but must not be nan!";
        *result = raise_domain_error<T_y>(function,msg_str.c_str(),y,Policy());
        return false;
      }
      return true;
    }

    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const std::vector<T_y>& y,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      for (int i = 0; i < y.size(); i++) {
        if (boost::math::isnan(y[i])) {
          std::ostringstream msg_o;
          msg_o << name << "[" << i << "] is %1%, but must not be nan!";
          *result = raise_domain_error<T_y>(function,msg_o.str().c_str(),y[i],Policy());
          return false;
        }
      }
      return true;
    }

    /**
     * Checks if the variable y is finite.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result,
                             const Policy& pol) {
      using boost::math::policies::raise_domain_error;
      if (!boost::math::isfinite(y)) {
        std::string message(name);
        message += " is %1%, but must be finite!";
        *result = raise_domain_error<T_y>(function,
                                          message.c_str(), 
                                          y, Policy());
        return false;
      }
      return true;
    }
    
    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const std::vector<T_y>& y,
                             const char* name,
                             T_result* result,
                             const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      for (int i = 0; i < y.size(); i++) {
        if (!boost::math::isfinite(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          *result = raise_domain_error<T_y>(function,
                                            message.str().c_str(),
                                            y[i], Policy());
          return false;
        }
      }
      return true;
    }


    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const char* name,  
                              T_result* result,
                              const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!boost::math::isfinite(x) || !(x > low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be finite and greater than "
            << low;
        *result = raise_domain_error<typename promote_args<T_x>::type>
                      (function, msg.str().c_str(), x, Policy());
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater_or_equal(const char* function,
                                       const T_x& x,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result,
                                       const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!boost::math::isfinite(x) || !(x >= low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be finite and greater than or equal to "
            << low;
        *result = raise_domain_error<typename promote_args<T_x>::type>
          (function, msg.str().c_str(), x, Policy());
        return false;
      }
      return true;
    }


    template <typename T_x, typename T_low, typename T_high, typename T_result,
              class Policy>
    inline bool check_bounded(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result,
                              const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      if (!boost::math::isfinite(x) || !(low <= x && x <= high)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be finite and between "
            << low
            << " and "
            << high;
        *result = raise_domain_error<typename boost::math::tools::promote_args<T_x>::type >
          (function,
           msg.str().c_str(),
           x, Policy());
        return false;
      }
      return true;
    }
    template <typename T_low, typename T_high, typename T_result, class Policy>
    inline bool check_bounded(const char* function,
                              const unsigned int x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result,
                              const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      if (!(low <= x && x <= high)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be finite and between "
            << low
            << " and "
            << high;
        *result = raise_domain_error<double>(function,
                                             msg.str().c_str(),
                                             x, Policy());
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_result, class Policy>
    inline bool check_nonnegative(const char* function,
                                  const T_x& x,
                                  const char* name,
                                  T_result* result,
                                  const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      if (!boost::math::isfinite(x) || !(x >= 0)) {
        std::string message(name);
        message += " is %1%, but must be finite and >= 0!";
        *result = raise_domain_error<typename boost::math::tools::promote_args<T_x>::type >(function,
                                                                                            message.c_str(), 
                                                                                            x, Policy());
        return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_nonnegative(const char* function,
                                  const unsigned int& x,
                                  const char* name,
                                  T_result* result,
                                  const Policy& /*pol*/) {
      return true;
    }

    template <bool finite=true,typename T_x, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const T_x& x,
                               const char* name,
                               T_result* result,
                               const Policy& /*pol*/) {
      if ((finite && !boost::math::isfinite(x))
          || (!finite && !boost::math::isnan(x))
          || !(x > 0)) {
        std::string message(name);
        message += " is %1%, but must be ";
        if (finite)
          message += "finite and ";
        message += "> 0!";
        *result = boost::math::policies::raise_domain_error<T_x>(function,
                                                                 message.c_str(), 
                                                                 x, Policy());
        
        return false;
      }
      return true;
    }
    
    template <bool finite=true,typename T_y, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const std::vector<T_y>& y,
                               const char* name,
                               T_result* result,
                               const Policy& /*pol*/) {
      for (int i = 0; i < y.size(); i++) {
        if ((finite && !boost::math::isfinite(y[i]))
            || (!finite && !boost::math::isnan(y[i]))
            || !(y[i] > 0)) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be ";
          if (finite)
            message << "finite and ";
          message << "> 0!";
          *result = boost::math::policies::raise_domain_error<T_y>(function,
                                                                   message.str().c_str(),
                                                                   y[i], Policy());
          return false;
        }
      }
      return true;
    }

    template <typename T_prob_vector, typename T_result, class Policy>
    inline bool check_simplex(const char* function,
                              const T_prob_vector& theta,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      if (theta.size() == 0) {
        std::string message(name);
        message += " is not a valid simplex. %1% elements in the vector.";
        *result = raise_domain_error<double>(function,message.c_str(),theta.size(),Policy());
      }
      if (fabs(1.0 - theta.sum()) > CONSTRAINT_TOLERANCE) {
        std::string message(name);
        message += " is not a valid simplex. The sum of the elements is %1%, but should be 1.0";
        *result = raise_domain_error(function, message.c_str(), theta.sum(), Policy());
        return false;
      }
      for (int n = 0; n < theta.size(); n++) {
        if (boost::math::isnan(theta[n]) || !(theta[n] >= 0)) {
          std::ostringstream stream;
          stream << name << " is not a valid simplex. The element at " << n << " is %1%, but should be greater than or equal to 0";
          *result = raise_domain_error(function, stream.str().c_str(), theta[n], Policy());
          return false;
        }
      }

      return true;
    }                         
    
  }
}

#endif


