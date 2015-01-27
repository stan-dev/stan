#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SQUARE_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is square.
     *
     * This check allows 0x0 matrices.
     *
     * @tparam T Type of scalar.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is a square matrix.
     * @throw <code>std::invalid_argument</code> if the matrix
     *    is not square
     */
    template <typename T_y>
    inline bool check_square(const char* function,
                             const char* name,
                             const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y) {
      check_size_match(function, 
                       "Expecting a square matrix; rows of ", name, y.rows(), 
                       "columns of ", name, y.cols());
      return true;
    }

  }
}
#endif
