#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/scalar/check_less_or_equal.hpp>
#include <stan/error_handling/matrix/check_lower_triangular.hpp>


namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor.  A Cholesky factor is a lower triangular
     * matrix whose diagonal elements are all positive.  Note that
     * Cholesky factors need not be square, but require at least as
     * many rows M as columns N (i.e., M &gt;= N).
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @return throws if any element in matrix is nan
     * @tparam T_y Type of elements of Cholesky factor
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
