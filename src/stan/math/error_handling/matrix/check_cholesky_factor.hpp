#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_CHOLESKY_FACTOR_HPP__

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
      if (!check_less_or_equal(function,y.cols(),y.rows(),
                               "columns and rows of Cholesky factor",
                               result))
        return false;
      if (!check_positive(function, y.cols(), "columns of Cholesky factor", 
                          result))
        return false;
      // FIXME:  should report row i
      if (!check_lower_triangular(function, y, name, result))
        return false;
      for (int i = 0; i < y.cols(); ++i)
        if (!check_positive(function, y(i,i), name, result))
          return false;
      return true;
    }

    template <typename T>
    inline bool check_cholesky_factor(const char* function,
                      const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                      const char* name,
                      T* result = 0) {
      return check_cholesky_factor<T,T>(function,y,name,result);
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
    inline bool check_cholesky_factor(const char* function,
                    const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                    T_result* result) {
      return check_cholesky_factor(function,y,"(internal variable)",result);
    }

    template <typename T>
    inline bool check_cholesky_factor(const char* function,
                    const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                    T* result = 0) {
      return check_cholesky_factor<T,T>(function,y,result);
    }

  }
}
#endif
