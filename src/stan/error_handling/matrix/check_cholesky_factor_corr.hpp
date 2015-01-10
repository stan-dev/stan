#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/matrix/check_lower_triangular.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/error_handling/matrix/check_unit_vector.hpp>

namespace stan {

  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor of a correlation matrix.
     *
     * A Cholesky factor is a lower triangular matrix whose diagonal
     * elements are all positive.  Note that Cholesky factors need not
     * be square, but require at least as many rows M as columns N
     * (i.e., M &gt;= N).
     *
     * Tolerance is specified by <code>math::CONSTRAINT_TOLERANCE</code>.
     *
     * @tparam T_y Type of elements of Cholesky factor
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     * @return <code>true</code> if the matrix is a valid Cholesky factor 
     *   of a correlation matrix
     * @throw <code>std::domain_error</code> if y is not a valid Choleksy factor,
     *   if number of rows is less than the number of columns, 
     *   if there are 0 columns,
     *   or if any element in matrix is NaN
     */
    template <typename T_y>
    bool check_cholesky_factor_corr(const std::string& function,
                                    const std::string& name,
                                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_square(function, name, y);
      check_lower_triangular(function, name, y);
      for (int i = 0; i < y.rows(); ++i)
        check_positive(function, name, y(i,i));
      for (int i = 0; i < y.rows(); ++i) {
        Eigen::Matrix<T_y,Eigen::Dynamic,1> 
          y_i = y.row(i).transpose();
        check_unit_vector(function, name, y_i);
      }
      return true;
    }

  }
}
#endif
