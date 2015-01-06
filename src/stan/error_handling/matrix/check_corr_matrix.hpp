#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CORR_MATRIX_HPP

#include <sstream>

#include <stan/error_handling/domain_error.hpp>
#include <stan/error_handling/scalar/check_positive_index.hpp>
#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>


namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * correlation matrix.  
     *
     * A valid correlation matrix is symmetric, has a unit diagonal
     * (all 1 values), and has all values between -1 and 1
     * (inclusive).
     *
     * This function throws exceptions if the variable is not a valid
     * correlation matrix.
     *
     * @tparam T_y Type of scalar
     *
     * @param function Name of the function this was called from
     * @param name Name of the variable
     * @param y Matrix to test
     * 
     * @return <code>true</code> if the specified matrix is a valid
     * correlation matrix
     * @throw <code>std::invalid_argument</code> if the matrix is not square
     *   or if the matrix is 0x0
     * @throw <code>std::domain_error</code> if the matrix is non-symmetric,
     *   diagonals not near 1, or not positive definite.
     */
    template <typename T_y>
    inline bool check_corr_matrix(const std::string& function,
                                  const std::string& name,
                                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;

      typedef typename index_type<Matrix<T_y,Dynamic,Dynamic> >::type size_t;

      check_size_match(function, 
                       "Rows of correlation matrix", y.rows(), 
                       "columns of correlation matrix", y.cols());
      
      check_positive_index(function, name, "rows", y.rows());

      check_symmetric(function, "y", y);
      
      for (size_t k = 0; k < y.rows(); ++k) {
        if (!(fabs(y(k,k) - 1.0) <= CONSTRAINT_TOLERANCE)) {
          std::ostringstream msg;
          msg << "is not a valid correlation matrix. " 
              << name << "(" << stan::error_index::value + k 
              << "," << stan::error_index::value + k 
              << ") is "; ;
          domain_error(function, name, y(k,k),
                  msg.str(),
                  ", but should be near 1.0");
          return false;
        }
      }
      stan::error_handling::check_pos_definite(function, "y", y);
      return true;
    }

  }
}
#endif
