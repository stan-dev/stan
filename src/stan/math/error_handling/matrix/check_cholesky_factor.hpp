#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/check_less_or_equal.hpp>
#include <stan/math/error_handling/matrix/check_lower_triangular.hpp>


namespace stan {
  namespace math {

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
     * @param result
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @tparam T_y Type of elements of Cholesky factor
     * @tparam T_result Type of result.
     */
    template <typename T_y, typename T_result>
    inline bool check_cholesky_factor(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result) {
      check_less_or_equal(function,y.cols(),y.rows(),
                          "columns and rows of Cholesky factor",
                          result);
      check_positive(function, y.cols(), "columns of Cholesky factor", 
                     result);
      // FIXME:  should report row i
      check_lower_triangular(function, y, name, result);
      for (int i = 0; i < y.cols(); ++i)
        check_positive(function, y(i,i), name, result);
      return true;
    }

  }
}
#endif
