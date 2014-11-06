#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SPSD_MATRIX_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/error_handling/matrix/check_pos_semidefinite.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if the specified matrix is a 
     * square, symmetric, and positive semi-definite.
     *
     * @param function
     * @param name
     * @param y Matrix to test.
     * @return <code>true</code> if the matrix is a square, symmetric,
     * and positive semi-definite.
     * @return throws if any element in matrix is nan.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y>
    inline bool check_spsd_matrix(const std::string& function, 
                                  const std::string& name,
                                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_square(function, name, y);
      check_positive(function, "rows", y.rows());
      check_symmetric(function, name, y);
      check_pos_semidefinite(function, name, y);
      return true;
    }

  }
}
#endif
