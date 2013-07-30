#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP__

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>


namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is square.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is a square matrix.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result>
    inline bool check_square(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result) {
      if (!check_size_match(function, 
                            y.rows(), "Rows of matrix",
                            y.cols(), "columns of matrix",
                            result))
        return false;
      return true;
    }

    template <typename T>
    inline bool check_square(const char* function,
                             const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                             const char* name,
                             T* result = 0) {
      return check_square<T,T>(function,y,name,result);
    }
  }
}
#endif
