#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP__

#include <sstream>
#include <boost/type_traits/common_type.hpp>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is lower
     * triangular.  A matrix x is not lower triangular if there is
     * a non-zero entry x[m,n] with m &lt; n.
     * 
     * @param function 
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is symmetric.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_lower_triangular(const char* function,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T_result* result) {
      for (int n = 1; n < y.cols(); ++n) {
        for (int m = 0; m < n && m < y.rows(); ++m) {
          if (y(m,n) != 0) {
            std::stringstream msg;
            msg << name << " is not lower triangular;"
                << " " << name << "[" << m << "," << n << "]="
                << "%1%"; 
            std::string msg_string(msg.str());
            return dom_err(function,y(m,n),"",msg_string.c_str(),"",result);
          }
        }
      }
      return true;
    }

    template <typename T>
    inline bool check_lower_triangular(const char* function,
                const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T* result = 0) {
      return check_lower_triangular<T,T>(function,y,name,result);
    }

  }
}
#endif
