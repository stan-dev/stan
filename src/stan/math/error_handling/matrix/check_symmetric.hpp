#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP__

#include <sstream>
#include <boost/type_traits/common_type.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/raise_domain_error.hpp>
#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is symmetric
     * 
     * NOTE: squareness is not checked by this function
     *
     * @param function 
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is symmetric.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T_result* result,
                const Policy&) {
      typedef 
        typename Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type;
      size_type k = y.rows();
      if (k == 1)
        return true;
      for (size_type m = 0; m < k; ++m) {
        for (size_type n = m + 1; n < k; ++n) {
          if (fabs(y(m,n) - y(n,m)) > CONSTRAINT_TOLERANCE) {
            std::ostringstream message;
            message << name << " is not symmetric. " 
                    << name << "[" << m << "," << n << "] is %1%, but "
                    << name << "[" << n << "," << m 
                    << "] element is " << y(n,m);
            T_result tmp 
              = policies::raise_domain_error<T_y>(function,
                                                  message.str().c_str(),
                                                  y(m,n), Policy());
            if (result != 0)
              *result = tmp;
            return false;
          }
        }
      }
      return true;
    }


    template <typename T_y, typename T_result>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T_result* result) {
      return check_symmetric(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T* result = 0) {
      return check_symmetric(function,y,name,result,default_policy());
    }

  }
}
#endif
