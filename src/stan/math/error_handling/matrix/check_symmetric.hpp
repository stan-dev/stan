#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SYMMETRIC_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/meta/traits.hpp>
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
    template <typename T_y, typename T_result>
    inline bool check_symmetric(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                                const char* name,
                                T_result* result) {
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
                    << name << "[" << stan::error_index::value + m << "," 
                    << stan::error_index::value +n << "] is %1%, but "
                    << name << "[" << stan::error_index::value +n << "," 
                    << stan::error_index::value + m 
                    << "] element is " << y(n,m);
            std::string msg(message.str());
            return dom_err(function,y(m,n),name,
                           msg.c_str(),"",
                           result);
          }
        }
      }
      return true;
    }

  }
}
#endif
