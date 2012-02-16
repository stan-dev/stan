#ifndef __STAN__MATH__ERROR_HANDLING_HPP__
#define __STAN__MATH__ERROR_HANDLING_HPP__

#include <stan/math/boost_error_handling.hpp>
#include <stan/math/special_functions.hpp>

#include <boost/math/policies/policy.hpp>
#include <cstddef>
#include <limits>


namespace stan { 

  namespace math {
    const double CONSTRAINT_TOLERANCE = 1E-8;


    /**
     * Default error-handling policy from Boost.
     */
    typedef boost::math::policies::policy<> default_policy;
    

    /**
     * Checks if the variable y is nan.
     *
     * @param function Name of function being invoked.
     * @param y Reference to variable being tested.
     * @param name Name of variable being tested.
     * @param result Pointer to resulting value after test.
     * @tparam T_y Type of variable being tested.
     * @tparam T_result Type of result returned.
     * @tparam Policy Error handling policy.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const T_y& y,
                              const char* name,
                              T_result* result,
                              const Policy& /* pol */) {
      if ((boost::math::isnan)(y)) {
        using stan::math::policies::raise_domain_error;
        std::string msg_str(name);
        msg_str += " is %1%, but must not be nan!";
        *result = raise_domain_error<T_result,T_y>(function,
                                                   msg_str.c_str(),
                                                   y,
                                                   Policy());
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
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if ((boost::math::isnan)(y[i])) {
          std::ostringstream msg_o;
          msg_o << name << "[" << i << "] is %1%, but must not be nan!";
          *result = raise_domain_error<T_result,T_y>(function,
                                                     msg_o.str().c_str(),
                                                     y[i],
                                                     Policy());
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
      using stan::math::policies::raise_domain_error;
      if (!(boost::math::isfinite)(y)) {
        std::string message(name);
        message += " is %1%, but must be finite!";
        *result = raise_domain_error<T_result,T_y>(function,
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
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if (!(boost::math::isfinite)(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          *result = raise_domain_error<T_result,T_y>(function,
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
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x > low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be greater than "
            << low;
        *result = raise_domain_error<T_result,T_x>(function, 
                                                   msg.str().c_str(), 
                                                   x, 
                                                   Policy());
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
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x >= low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be greater than or equal to "
            << low;
        *result = raise_domain_error<T_result,T_x>(function, 
                                                   msg.str().c_str(), 
                                                   x, 
                                                   Policy());
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
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(low <= x && x <= high)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be between "
            << low
            << " and "
            << high;
        *result = raise_domain_error<T_result,T_x>(function,
                                                   msg.str().c_str(),
                                                   x, 
                                                   Policy());
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
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (std::numeric_limits<T_x>::is_signed && !(x >= 0)) {
        std::string message(name);
        message += " is %1%, but must be >= 0!";
        *result = raise_domain_error<T_result,T_x>(function,
                                                   message.c_str(), 
                                                   x, 
                                                   Policy());
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const T_x& x,
                               const char* name,
                               T_result* result,
                               const Policy& /*pol*/) {
      using stan::math::policies::raise_domain_error;
      if (!(x > 0)) {
        std::string message(name);
        message += " is %1%, but must be > 0";
        *result = raise_domain_error<T_result,T_x>(function,
                                                   message.c_str(), 
                                                   x, Policy());
        
        return false;
      }
      return true;
    }
    
    template <typename T_y, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const std::vector<T_y>& y,
                               const char* name,
                               T_result* result,
                               const Policy& /*pol*/) {
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if (!(y[i] > 0)) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be > 0";
          *result = raise_domain_error<T_result,T_y>(function,
                                                     message.str().c_str(),
                                                     y[i], 
                                                     Policy());
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
      using stan::math::policies::raise_domain_error;
      typename T_prob_vector::value_type T_prob;
      if (theta.size() == 0) {
        std::string message(name);
        message += " is not a valid simplex. %1% elements in the vector.";
        *result = raise_domain_error<T_result,T_prob>(function,
                                                      message.c_str(),
                                                      theta.size(),
                                                      Policy());
      }
      if (fabs(1.0 - theta.sum()) > CONSTRAINT_TOLERANCE) {
        std::string message(name);
        message += " is not a valid simplex.";
        message += " The sum of the elements is %1%, but should be 1.0";
        *result = raise_domain_error<T_result,T_prob>(function, 
                                                      message.c_str(), 
                                                      theta.sum(), 
                                                      Policy());
        return false;
      }
      for (size_t n = 0; n < theta.size(); n++) {
        if ((boost::math::isnan)(theta[n]) || !(theta[n] >= 0)) {
          std::ostringstream stream;
          stream << name << " is not a valid simplex."
                 << " The element at " << n 
                 << " is %1%, but should be greater than or equal to 0";
          *result = raise_domain_error<T_result,T_prob>(function, 
                                                        stream.str().c_str(), 
                                                        theta[n], 
                                                        Policy());
          return false;
        }
      }
      return true;
    }                         
    
  }
}

#endif


