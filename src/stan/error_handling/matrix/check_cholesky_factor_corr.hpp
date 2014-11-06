#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/matrix/check_lower_triangular.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/error_handling/matrix/check_unit_vector.hpp>

namespace stan {

  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor.  A Cholesky factor is a lower triangular
     * matrix whose diagonal elements are all positive.  Note that
     * Cholesky factors need not be square, but require at least as
     * many rows M as columns N (i.e., M &gt;= N).
     *
     * Tolerance is specified by <code>math::CONSTRAINT_TOLERANCE</code>.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @return throws if any element in y is nan.
     * @tparam T_y Type of elements of Cholesky factor
     */
    template <typename T_y>
    bool check_cholesky_factor_corr(const std::string& function,
                                    const std::string& name,
                                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      if (!check_square(function, name, y))
        return false;
      if (!check_lower_triangular(function, name, y))
        return false;
      for (int i = 0; i < y.rows(); ++i)
        if (!check_positive(function, name, y(i,i)))
          return false;
      for (int i = 0; i < y.rows(); ++i) {
        Eigen::Matrix<T_y,Eigen::Dynamic,1> 
          y_i = y.row(i).transpose();
        if (!check_unit_vector(function, name, y_i))
          return false;
      }
      return true;
    }


    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor (lower triangular, positive diagonal).
     *
     * @param function Name of function.
     * @param y Matrix to test.
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @return throws if any element in matrix is nan
     * @tparam T_y Type of elements of Cholesky factor
     * @tparam T_result Type of result.
     */
    template <typename T_y>
    inline bool check_cholesky_factor_corr(const std::string& function,
                                           const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      return check_cholesky_factor_corr(function, "(internal variable)", y);
    }

  }
}
#endif
