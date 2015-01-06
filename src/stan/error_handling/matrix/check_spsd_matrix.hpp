#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive_index.hpp>
#include <stan/error_handling/matrix/check_pos_semidefinite.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a 
     * square, symmetric, and positive semi-definite.
     *
     * @tparam T Scalar type of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is a square, symmetric,
     *   and positive semi-definite.
     * @throw <code>std::invalid_argument</code> if the matrix is not square
     *   or if the matrix is 0x0
     * @throw <code>std::domain_error</code> if the matrix is not symmetric
     *   or if the matrix is not positive semi-definite
     */
    template <typename T_y>
    inline bool check_spsd_matrix(const std::string& function, 
                                  const std::string& name,
                                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_square(function, name, y);
      check_positive_index(function, name, "rows()", y.rows());
      check_symmetric(function, name, y);
      check_pos_semidefinite(function, name, y);
      return true;
    }

  }
}
#endif
