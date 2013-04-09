#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_POSITIVE_ORDERED_HPP__

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/raise_domain_error.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified vector contains
     * only non-negative values and is sorted into increasing order.
     * There may be duplicate values.  Otherwise, raise a domain
     * error according to the specified policy.
     *
     * @param function
     * @param y Vector to test.
     * @param name
     * @param result
     * @tparam Policy Only the policy's type matters.
     * @return <code>true</code> if the vector has positive, ordered
     * values.
     */
    template <typename T_y, typename T_result, class Policy>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                                const char* name,
                                T_result* result,
                                const Policy&) {
      using stan::math::policies::raise_domain_error;
      typedef typename Eigen::Matrix<T_y,Eigen::Dynamic,1>::size_type size_t;
      if (y.size() == 0) {
        return true;
      }
      if (y[0] < 0) {
        std::ostringstream stream;
        stream << name << " is not a valid positive_ordered vector."
               << " The element at 0 is %1%, but should be postive.";
        T_result tmp = raise_domain_error<T_result,T_y>(function, 
                                                        stream.str().c_str(), 
                                                        y[0], 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      for (size_t n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream stream;
          stream << name << " is not a valid ordered vector."
                 << " The element at " << n 
                 << " is %1%, but should be greater than the previous element, "
                 << y[n-1];
          T_result tmp = raise_domain_error<T_result,T_y>(function, 
                                                          stream.str().c_str(), 
                                                          y[n], 
                                                          Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }                         
    template <typename T_y, typename T_result>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                                const char* name,
                                T_result* result) {
      return check_positive_ordered(function,y,name,result,default_policy());
    }
    template <typename T>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                                const char* name,
                                T* result = 0) {
      return check_positive_ordered(function,y,name,result,default_policy());
    }

  }
}
#endif
