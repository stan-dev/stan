#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  A valid covariance matrix must be square,
     * symmetric, and positive definite.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is a valid covariance matrix.
     * @return throws if any element in matrix is nan
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y>
    inline bool check_cov_matrix(const std::string& function,
                                 const std::string& name,
                                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_size_match(function, 
                       "Rows of covariance matrix", y.rows(),
                       "columns of covariance matrix", y.cols());
      check_positive(function, "rows", y.rows());
      check_symmetric(function, name, y);
      check_pos_definite(function, name, y);
      return true;
    }

  }
}
#endif
