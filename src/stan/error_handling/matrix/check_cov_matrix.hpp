#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_pos_definite.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  
     *
     * A valid covariance matrix is a square, symmetric matrix that is
     * positive definite.
     *
     * @tparam T Type of scalar.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is a valid covariance matrix
     * @throw <code>std::invalid_argument</code> if the matrix is not square
     *   or if the matrix is 0x0
     * @throw <code>std::domain_error</code> if the matrix is not symmetric, 
     *   if the matrix is not positive definite, 
     *   or if any element of the matrix is nan
     */
    template <typename T_y>
    inline bool check_cov_matrix(const std::string& function,
                                 const std::string& name,
                                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_pos_definite(function, name, y);
      return true;
    }

  }
}
#endif
