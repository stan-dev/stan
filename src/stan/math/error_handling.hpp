#ifndef __STAN__MATH__ERROR_HANDLING_HPP__
#define __STAN__MATH__ERROR_HANDLING_HPP__

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <iostream>

#include <boost/math/policies/policy.hpp>
#include <boost/type_traits/is_unsigned.hpp>

#include <stan/math/boost_error_handling.hpp>
#include <stan/meta/traits.hpp>

namespace stan { 

  namespace math {
    
    /**
     * The tolerance for checking arithmetic bounds In rank and in
     * simplexes.  The default value is <code>1E-8</code>.
     */
    const double CONSTRAINT_TOLERANCE = 1E-8;


    /**
     * Default error-handling policy from Boost.
     */
    typedef boost::math::policies::policy<> default_policy;

    namespace {
    
      // local output stream for pairs
      template <typename T1, typename T2>
      std::ostream& operator<<(std::ostream& o,
                               std::pair<T1,T2> xs) {
        o << '(' << xs.first << ", " << xs.second << ')';
        return o;
      }

      template <typename T_y, 
                typename T_result,
                typename T_msg2,
                class Policy>
      bool domain_error(const char* function,
                        const T_y& y,
                        const char* name,
                        const char* error_msg,
                        T_msg2 error_msg2,
                        T_result* result,
                        const Policy&) {
        using stan::math::policies::raise_domain_error;
        std::ostringstream msg_o;
        msg_o << name << error_msg << error_msg2;
        T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                        msg_o.str().c_str(),
                                                        y,
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }

      template <typename T_y, 
                typename T_result,
                typename T_msg2,
                class Policy>
      inline bool domain_error_vec(size_t i,
                                   const char* function,
                                   const std::vector<T_y>& y,
                                   const char* name,
                                   const char* error_msg,
                                   T_msg2 error_msg2,
                                   T_result* result,
                                   const Policy&) {
        using stan::math::policies::raise_domain_error;
        std::ostringstream msg_o;
        msg_o << name << "[" << i << "] " << error_msg << error_msg2;
        T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                        msg_o.str().c_str(),
                                                        y[i],
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
    } // end anon namespace



    
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
      if ((boost::math::isnan)(y)) 
        return domain_error(function,y,name,
                            " is %1%, but must not be nan!","",
                            result,Policy());
      return true;
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
      for (size_t i = 0; i < y.size(); i++)
        if ((boost::math::isnan)(y[i])) 
          return domain_error_vec(i,function,y,name,
                                  " is %1%, but must not be nan!","",
                                  result,Policy());
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
    // need this sig to infer types for result from type of y
    template <typename T>
    inline bool check_not_nan(const char* function,
                              const T& y,
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
      if (!(boost::math::isfinite)(y))
        return domain_error(function,y,name,
                            " is %1%, but must be finite!","",
                            result,Policy());
      return true;
    }
    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const std::vector<T_y>& y,
                             const char* name,
                             T_result* result,
                             const Policy&) {
      for (size_t i = 0; i < y.size(); i++) 
        if (!(boost::math::isfinite)(y[i])) 
          return domain_error_vec(i,function,y,name,
                                  " is %1%, but must be finite!","",
                                  result,Policy());
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




    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater(const char* function,
                              const T_x& x,
                              const T_low& low,
                              const char* name,  
                              T_result* result,
                              const Policy&) {

      if (!(x > low))
        return domain_error(function,x,name,
                            " is %1%, but must be greater than ",
                            low,result,Policy());
      return true;
    }
    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater(const char* function,
                              const std::vector<T_x>& x,
                              const T_low& low,
                              const char* name,  
                              T_result* result,
                              const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!(x[i] > low))
          return domain_error_vec(i,function,x,name,
                                  " is %1%, but must be greater than ",
                                  low,result,Policy());
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
      if (!(x >= low))
        return domain_error(function,x,name,
                            " is %1%, but must be greater than or equal to ",
                            low,result,Policy());
      return true;
    }
    template <typename T_x, typename T_low, typename T_result, class Policy>
    inline bool check_greater_or_equal(const char* function,
                                       const std::vector<T_x>& x,
                                       const T_low& low,
                                       const char* name,  
                                       T_result* result,
                                       const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!(x[i] >= low))
          return domain_error_vec(
                          i,function,x,name,
                          " is %1%, but must be greater than or equal to",
                          low,result,Policy());
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



    template <typename T_x, typename T_high, typename T_result, class Policy>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_high& high,
                           const char* name,  
                           T_result* result,
                           const Policy&) {
      if (!(x < high)) 
        return domain_error(function,x,name,
                            " is %1%, but must be less than ",
                            high,result,Policy());
      return true;
    }
    template <typename T_x, typename T_high, typename T_result, class Policy>
    inline bool check_less(const char* function,
                           const std::vector<T_x>& x,
                           const T_high& high,
                           const char* name,  
                           T_result* result,
                           const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!(x[i] < high))
          return domain_error_vec(i,function,x,name,
                                  " is %1%, but must be less than",
                                  high,result,Policy());
      return true;
    }
    template <typename T_x, typename T_high, typename T_result>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_high& high,
                           const char* name,  
                           T_result* result) {
      return check_less(function,x,high,name,result,default_policy());
    }
    template <typename T_x, typename T_high>
    inline bool check_less(const char* function,
                           const T_x& x,
                           const T_high& high,
                           const char* name,  
                           T_x* result = 0) {
      return check_less(function,x,high,name,result,default_policy());
    }



    template <typename T_x, typename T_high, typename T_result, class Policy>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_high& high,
                                    const char* name,  
                                    T_result* result,
                                    const Policy&) {
      if (!(x <= high))
        return domain_error(function,x,name,
                            " is %1%, but must be less than or equal to ",
                            high,result,Policy());
      return true;
    }
    template <typename T_x, typename T_high, typename T_result, class Policy>
    inline bool check_less_or_equal(const char* function,
                                    const std::vector<T_x>& x,
                                    const T_high& high,
                                    const char* name,  
                                    T_result* result,
                                    const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!(x[i] <= high))
          return domain_error_vec(
                                  i,function,x,name,
                                  " is %1%, but must be less than or equal to",
                                  high,result,Policy());
      return true;
    }
    template <typename T_x, typename T_high, typename T_result>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_high& high,
                                    const char* name,  
                                    T_result* result) {
      return check_less_or_equal(function,x,high,name,result,default_policy());
    }
    template <typename T_x, typename T_high>
    inline bool check_less_or_equal(const char* function,
                                    const T_x& x,
                                    const T_high& high,
                                    const char* name,  
                                    T_x* result = 0) {
      return check_less_or_equal(function,x,high,name,result,default_policy());
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
      if (!(low <= x && x <= high))
        return domain_error(function,x,name," is %1%, but must be between ",
                            std::pair<T_low,T_high>(low,high),
                            result,Policy());
      return true;
    }
    template <typename T_x, typename T_low, typename T_high, typename T_result,
              class Policy>
    inline bool check_bounded(const char* function,
                              const std::vector<T_x>& x,
                              const T_low& low,
                              const T_high& high,
                              const char* name,  
                              T_result* result,
                              const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!(low <= x[i] && x[i] <= high))
          return domain_error_vec(i,function,x,name,
                                  " is %1%, but must be between ",
                                  std::pair<T_low,T_high>(low,high),
                                  result,Policy());
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
      // have to use not is_unsigned. is_signed will be false
      // floating point types that have no unsigned versions.
      if (!boost::is_unsigned<T_x>::value && !(x >= 0)) 
        return domain_error(function,x,name,
                            " is %1%, but must be >= 0!","",result,Policy());
      return true;
    }
    template <typename T_x, typename T_result, 
              class Policy>
    inline bool check_nonnegative(const char* function,
                                  const std::vector<T_x>& x,
                                  const char* name,
                                  T_result* result,
                                  const Policy&) {
      for (size_t i = 0; i < x.size(); ++i)
        if (!boost::is_unsigned<T_x>::value && !(x[i] >= 0)) 
          return domain_error_vec(i,function,x,name,
                                  " is %1%, but must be >= 0!","",
                                  result,Policy());
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
      if (!(x > 0))
        return domain_error(function,x,name,
                            " is %1%, but must be > 0","",result,Policy());
      return true;
    }
    template <typename T_y, typename T_result, class Policy>
    inline bool check_positive(const char* function,
                               const std::vector<T_y>& y,
                               const char* name,
                               T_result* result,
                               const Policy&) { 
      for (size_t i = 0; i < y.size(); i++) 
        if (!(y[i] > 0)) 
          return domain_error_vec(i,function,y,name,
                                  " is %1%, but must be > 0","",
                                  result,Policy());
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

    template <typename T>
    size_t size_of(const std::vector<T>& x) {
      return x.size();
    }
    template <typename T>
    size_t size_of(T x) {
      return 1U;
    }

    template <typename T, typename T_result, class Policy>
    inline bool check_consistent_size(size_t max_size,
                                      const char* function,
                                      const T& x,
                                      const char* name,
                                      T_result* result,
                                      const Policy&) {
      size_t x_size = size_of(x);
      if (x_size == 1 || x_size == max_size)
        return true;
      return domain_error(
              function,x_size,name,
              " (max size) is %1%, but must be consistent, 1 or max=",max_size,
              result,Policy());
    }

    template <typename T1, typename T2, typename T3, typename T_result, 
              class Policy>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       T_result* result,
                                       const Policy&) {
      size_t max_size = std::max(size_of(x1),
                                 std::max(size_of(x2),size_of(x3)));
      return check_consistent_size(max_size,function,x1,name1,result,Policy())
        && check_consistent_size(max_size,function,x2,name2,result,Policy())
        && check_consistent_size(max_size,function,x3,name3,result,Policy());
    }
    template <typename T1, typename T2, typename T3, typename T_result>
    inline bool check_consistent_sizes(const char* function,
                                       const T1& x1, 
                                       const T2& x2, 
                                       const T3& x3,
                                       const char* name1,
                                       const char* name2,
                                       const char* name3,
                                       T_result* result) {
      return check_consistent_sizes(function,x1,x2,x3,name1,name2,name3,
                                    result,default_policy());
    }
    template <typename T1, typename T2, typename T3, typename T_result>
    inline bool check_consistent_sizes(
                       const char* function,
                       const T1& x1, 
                       const T2& x2, 
                       const T3& x3,
                       const char* name1,
                       const char* name2,
                       const char* name3,
                       typename stan::return_type<T1,T2,T3>::type* result) {
      return check_consistent_sizes(function,x1,x2,x3,name1,name2,name3,
                                    result,default_policy());
    }
    

  }
}

#endif

