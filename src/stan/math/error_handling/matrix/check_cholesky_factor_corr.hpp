#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_CORR_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/matrix/check_lower_triangular.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/error_handling/matrix/check_unit_vector.hpp>

namespace stan {

  namespace math {

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
     * @param result
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @tparam T_y Type of elements of Cholesky factor
     * @tparam T_result Type of result.
     */
    template <typename T_y, typename T_result>
    bool check_cholesky_factor_corr(const char* function,
                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                    const char* name,
                    T_result* result) {
      if (!check_square(function,y,name,result))
        return false;
      if (!check_lower_triangular(function,y,name,result))
        return false;
      for (int i = 0; i < y.rows(); ++i)
        if (!check_positive(function, y(i,i), name, result))
          return false;
      for (int i = 0; i < y.rows(); ++i) {
        Eigen::Matrix<T_y,Eigen::Dynamic,1> y_i = y.row(i).transpose();
        if (!check_unit_vector(function, y_i, name, result))
          return false;
      }
      return true;
    }

    template <typename T>
    inline bool check_cholesky_factor_corr(const char* function,
                      const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                      const char* name,
                      T* result = 0) {
      return check_cholesky_factor_corr<T,T>(function,y,name,result);
    }


    /**
     * Return <code>true</code> if the specified matrix is a valid
     * Cholesky factor (lower triangular, positive diagonal).
     *
     * @param function Name of function.
     * @param y Matrix to test.
     * @param result Pointer into which to write result.
     * @return <code>true</code> if the matrix is a valid Cholesky factor.
     * @tparam T_y Type of elements of Cholesky factor
     * @tparam T_result Type of result.
     */
    template <typename T_y, typename T_result>
    inline bool check_cholesky_factor_corr(const char* function,
                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                    T_result* result) {
      return check_cholesky_factor_corr(function,y,"(internal variable)",
                                        result);
    }

    template <typename T>
    inline bool check_cholesky_factor_corr(const char* function,
                    const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                    T* result = 0) {
      return check_cholesky_factor_corr<T,T>(function,y,result);
    }

  }
}
#endif
