#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_LOWER_TRIANGULAR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/meta/traits.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is lower
     * triangular.  A matrix x is not lower triangular if there is
     * a non-zero entry x[m,n] with m &lt; n.
     * 
     * @param function 
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is symmetric.
     * @return throws if any element in upper triangular is nan
     * @tparam T Type of scalar.
     */
    template <typename T_y>
    inline bool check_lower_triangular(const std::string& function,
                const std::string& name,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      for (int n = 1; n < y.cols(); ++n) {
        for (int m = 0; m < n && m < y.rows(); ++m) {
          if (y(m,n) != 0) {
            std::stringstream msg;
            msg << "is not lower triangular;"
                << " " << name << "[" << stan::error_index::value + m << "," 
                << stan::error_index::value + n << "]=";
            dom_err(function, name, y(m,n),
                    msg.str());
            return false;
          }
        }
      }
      return true;
    }

  }
}
#endif
