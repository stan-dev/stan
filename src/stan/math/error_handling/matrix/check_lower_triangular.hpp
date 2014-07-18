#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/error_handling/dom_err.hpp>

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
                << " " << name << "[" << stan::error_index::value + m << "," 
                << stan::error_index::value + n << "]="
                << "%1%"; 
            std::string msg_string(msg.str());
            return dom_err(function,y(m,n),"",msg_string.c_str(),"",result);
          }
        }
      }
      return true;
    }

  }
}
#endif
