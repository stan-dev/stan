#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>
#include <stan/error_handling/matrix/check_lower_triangular.hpp>


namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor.  
     *
     * A Cholesky factor is a lower triangular matrix whose diagonal
     * elements are all positive.  Note that Cholesky factors need not
     * be square, but require at least as many rows M as columns N
     * (i.e., M &gt;= N).
     *
     * @tparam T_y Type of elements of Cholesky factor
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is a valid Cholesky factor
     * @throw <code>std::domain_error</code> if y is not a valid Choleksy factor,
     *   if number of rows is less than the number of columns, 
     *   if there are 0 columns,
     *   or if any element in matrix is NaN
     */
    template <typename T_y>
    inline bool check_cholesky_factor(const std::string& function,
                                      const std::string& name,
                                      const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_less_or_equal(function, "columns and rows of Cholesky factor",
                          y.cols(), y.rows());
      check_positive(function, "columns of Cholesky factor", y.cols());
      // FIXME:  should report row i
      check_lower_triangular(function, name, y);
      for (int i = 0; i < y.cols(); ++i)
        check_positive(function, name, y(i,i));
      return true;
    }

  }
}
#endif
