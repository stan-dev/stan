#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP

#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <sstream>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"

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
      check_size_match(function, 
                       y.rows(), "Rows of matrix",
                       y.cols(), "columns of matrix",
                       result);
      return true;
    }

  }
}
#endif
