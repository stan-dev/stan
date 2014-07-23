#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/check_positive.hpp>
#include <stan/math/error_handling/matrix/check_pos_semidefinite.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is a 
     * square, symmetric, and positive semi-definite.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is a square, symmetric,
     * and positive semi-definite.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y, typename T_result>
    inline bool check_spsd_matrix(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result) {
      check_square(function, y, name, result);
      check_positive(function, y.rows(), "rows", result);
      check_symmetric(function, y, name, result);
      check_pos_semidefinite(function, y, name, result);
      return true;
    }

  }
}
#endif
