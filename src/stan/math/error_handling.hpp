#ifndef __STAN__MATH__ERROR_HANDLING_HPP__
#define __STAN__MATH__ERROR_HANDLING_HPP__

#include <stan/math/boost_error_handling.hpp>

#include <boost/math/policies/policy.hpp>
#include <cstddef>
#include <limits>


namespace stan { 

  namespace math {
    /**
     * This is the tolerance for checking arithmetic bounds
     * In rank and in simplexes.  The current value is <code>1E-8</code>.
     */
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
    template <typename T_y, 
              typename T_result,
              class Policy>
    inline bool check_not_nan(const char* function,
                              const T_y& y,
                              const char* name,
                              T_result* result,
                              const Policy&) {
      if ((boost::math::isnan)(y)) {
        using stan::math::policies::raise_domain_error;
        std::string msg_str(name);
        msg_str += " is %1%, but must not be nan!";
        T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                        msg_str.c_str(),
                                                        y,
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_y, 
              typename T_result>
    inline bool check_not_nan(const char* function,
                              const T_y& y,
                              const char* name,
                              T_result* result = 0) {
      return check_not_nan(function,y,name,result,default_policy());
    }

    // need this sig to infer types for result
    template <typename T>
    inline bool check_not_nan(const char* function,
                              const T& y,
                              const char* name,
                              T* result = 0) {
      return check_not_nan(function,y,name,result,default_policy());
    }




    /**
     * Check that the specified argument vector does not contain a nan.
     */
    template <typename T_y, 
              typename T_result,
              class Policy>
    inline bool check_not_nan(const char* function,
                              const std::vector<T_y>& y,
                              const char* name,
                              T_result* result,
                              const Policy&) {
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if ((boost::math::isnan)(y[i])) {
          std::ostringstream msg_o;
          msg_o << name << "[" << i << "] is %1%, but must not be nan!";
          T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                          msg_o.str().c_str(),
                                                          y[i],
                                                          Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }

    template <typename T_y, 
              typename T_result>
    inline bool check_not_nan(const char* function,
                              const std::vector<T_y>& y,
                              const char* name,
                              T_result* result) {
      return check_not_nan(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_not_nan(const char* function,
                              const std::vector<T>& y,
                              const char* name,
                              T* result = 0) {
      return check_not_nan(function,y,name,result,default_policy());
    }




    /**
     * Checks if the variable y is finite.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result,
                             const Policy&) {
      using stan::math::policies::raise_domain_error;
      if (!(boost::math::isfinite)(y)) {
        std::string message(name);
        message += " is %1%, but must be finite!";
        T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                        message.c_str(), 
                                                        y, Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_y, typename T_result>
    inline bool check_finite(const char* function,
                             const T_y& y,
                             const char* name,
                             T_result* result) {
      return check_finite(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_finite(const char* function,
                             const T& y,
                             const char* name,
                             T* result = 0) {
      return check_finite(function,y,name,result,default_policy());
    }




    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const std::vector<T_y>& y,
                             const char* name,
                             T_result* result,
                             const Policy&) {
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if (!(boost::math::isfinite)(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                          message.str().c_str(),
                                                          y[i], Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }

    template <typename T_y, typename T_result>
    inline bool check_finite(const char* function,
                             const std::vector<T_y>& y,
                             const char* name,
                             T_result* result) {
      return check_finite(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_finite(const char* function,
                             const std::vector<T>& y,
                             const char* name,
                             T* result = 0) {
      return check_finite(function,y,name,result,default_policy());
    }




    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const char* name,  
                              T_result* result,
                              const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x > low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be greater than "
            << low;
        T_result tmp = raise_domain_error<T_result,T_x>(function, 
                                                        msg.str().c_str(), 
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_result>
    inline bool check_greater(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const char* name,  
                              T_result* result) {
      return check_greater(function,x,low,name,result,default_policy());
    }

    template <typename T_x, typename T_low>
    inline bool check_greater(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const char* name,  
                              T_x* result = 0) {
      return check_greater(function,x,low,name,result,default_policy());
    }




    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater_or_equal(const char* function,
                                       const T_x& x,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result,
                                       const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x >= low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be greater or equal to "
            << low;
        T_result tmp = raise_domain_error<T_result,T_x>(function, 
                                                        msg.str().c_str(), 
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_result>
    inline bool check_greater_or_equal(const char* function,
                                       const T_x& x,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result) {
      return check_greater_or_equal(function,x,low,name,result,
                                    default_policy());
    }                               

    template <typename T_x, typename T_low>
    inline bool check_greater_or_equal(const char* function,
                                       const T_x& x,
                                       const T_low& low,
                                       const char* name,  
                                       T_x* result = 0) {
      return check_greater_or_equal(function,x,low,name,result,
                                    default_policy());
    }




    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_low& low,
                           const char* name,  
                           T_result* result,
                           const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x < low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be less than "
            << low;
        T_result tmp = raise_domain_error<T_result,T_x>(function, 
                                                        msg.str().c_str(), 
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_result>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_low& low,
                           const char* name,  
                           T_result* result) {
      return check_less(function,x,low,name,result,default_policy());
    }

    template <typename T_x, typename T_low>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_low& low,
                           const char* name,  
                           T_x* result = 0) {
      return check_less(function,x,low,name,result,default_policy());
    }



    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_low& low,
                                    const char* name,  
                                    T_result* result,
                                    const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(x <= low)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be less than or equal to "
            << low;
        T_result tmp = raise_domain_error<T_result,T_x>(function, 
                                                        msg.str().c_str(), 
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_result>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_low& low,
                                    const char* name,  
                                    T_result* result) {
      return check_less_or_equal(function,x,low,name,result,default_policy());
    }

    template <typename T_x, typename T_low>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_low& low,
                                    const char* name,  
                                    T_x* result = 0) {
      return check_less_or_equal(function,x,low,name,result,default_policy());
    }




    template <typename T_x, typename T_low, typename T_high, typename T_result,
              class Policy>
    inline bool check_bounded(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result,
                              const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      if (!(low <= x && x <= high)) {
        std::ostringstream msg;
        msg << name 
            << " is %1%, but must be between "
            << low
            << " and "
            << high;
        T_result tmp = raise_domain_error<T_result,T_x>(function,
                                                        msg.str().c_str(),
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_low, typename T_high, typename T_result>
    inline bool check_bounded(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result) {
      return check_bounded(function,x,low,high,name,result,default_policy());
    }

    template <typename T_x, typename T_low, typename T_high>
    inline bool check_bounded(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_x* result = 0) {
      return check_bounded(function,x,low,high,name,result,default_policy());
    }




    template <typename T_x, typename T_result, 
              class Policy>
    inline bool check_nonnegative(const char* function,
                                  const T_x& x,
                                  const char* name,
                                  T_result* result,
                                  const Policy&) {
      using stan::math::policies::raise_domain_error;
      using boost::math::tools::promote_args;
      // have to use not is_unsigned. is_signed will be false
      // floating point types that have no unsigned versions.
      if (!boost::is_unsigned<T_x>::value && !(x >= 0)) {
        std::string message(name);
        message += " is %1%, but must be >= 0!";
        T_result tmp = raise_domain_error<T_result,T_x>(function,
                                                        message.c_str(), 
                                                        x, 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_x, typename T_result>
    inline bool check_nonnegative(const char* function,
                                  const T_x& x,
                                  const char* name,
                                  T_result* result) {
      return check_nonnegative(function,x,name,result,default_policy());
    }

    template <typename T>
    inline bool check_nonnegative(const char* function,
                                  const T& x,
                                  const char* name,
                                  T* result = 0) {
      return check_nonnegative(function,x,name,result,default_policy());
    }




    template <typename T_x, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const T_x& x,
                               const char* name,
                               T_result* result,
                               const Policy&) {
      using stan::math::policies::raise_domain_error;
      if (!(x > 0)) {
        std::string message(name);
        message += " is %1%, but must be > 0";
        T_result tmp = raise_domain_error<T_result,T_x>(function,
                                                        message.c_str(), 
                                                        x, Policy());
        
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
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
                               const char* name,
                               T* result = 0) {
      return check_positive(function,x,name,result,default_policy());
    }




    template <typename T_y, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const std::vector<T_y>& y,
                               const char* name,
                               T_result* result,
                               const Policy&) { 
      using stan::math::policies::raise_domain_error;
      for (size_t i = 0; i < y.size(); i++) {
        if (!(y[i] > 0)) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be > 0";
          T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                          message.str().c_str(),
                                                          y[i], 
                                                          Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }

    template <typename T_y, typename T_result>
    inline bool check_positive(const char* function,
                               const std::vector<T_y>& y,
                               const char* name,
                               T_result* result) {
      return check_positive(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_positive(const char* function,
                               const std::vector<T>& y,
                               const char* name,
                               T* result = 0) {
      return check_positive(function,y,name,result,default_policy());
    }






  }
}

#endif

