#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_COV_MATRIX_HPP__

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>


namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  A valid covariance matrix must be square,
     * symmetric, and positive definite.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is a valid covariance matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y, typename T_result>
    inline bool check_cov_matrix(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result) {
      if (!check_size_match(function, 
                            y.rows(), "Rows of covariance matrix",
                            y.cols(), "columns of covariance matrix",
                            result)) 
        return false;
      if (!check_positive(function, y.rows(), "rows", result))
        return false;
      if (!check_symmetric(function, y, name, result))
        return false;
      if (!check_pos_definite(function, y, name, result))
        return false;
      return true;
    }

    template <typename T>
    inline bool check_cov_matrix(const char* function,
                                 const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                                 const char* name,
                                 T* result = 0) {
      return check_cov_matrix<T,T>(function,y,name,result);
    }


    // FIXME: this looks redundant
    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  A valid covariance matrix must be symmetric
     * and positive definite.
     *
     * @param function
     * @param Sigma Matrix to test.
     * @param result
     * @return <code>true</code> if the matrix is a valid covariance matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_covar, typename T_result>
    inline bool check_cov_matrix(const char* function,
                                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                                 T_result* result) {
      if (!check_size_match(function, 
                            Sigma.rows(), "Rows of covariance matrix",
                            Sigma.cols(), "columns of covariance matrix",
                            result)) 
        return false;
      if (!check_positive(function, Sigma.rows(), "rows", result))
        return false;
      if (!check_symmetric(function, Sigma, "Sigma", result))
        return false;
      return true;
    }
    template <typename T>
    inline bool check_cov_matrix(const char* function,
                                 const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                                 T* result = 0) {
      return check_cov_matrix<T,T>(function,Sigma,result);
    }

  }
}
#endif
