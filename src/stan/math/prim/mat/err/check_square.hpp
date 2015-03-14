#ifndef STAN__MATH__PRIM__MAT__ERR__CHECK_SQUARE_HPP
#define STAN__MATH__PRIM__MAT__ERR__CHECK_SQUARE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <sstream>

namespace stan {
  namespace math {

    /**
     * Return <code>true</code> if the specified matrix is square.
     *
     * This check allows 0x0 matrices.
     *
     * @tparam T Type of matrix.
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
    inline bool
    check_square(const char* function,
                 const char* name,
                 const Eigen::MatrixBase<T_y>& y) {
      check_size_match(function,
                       "Expecting a square matrix; rows of ", name, y.rows(),
                       "columns of ", name, y.cols());
      return true;
    }

  }
}
#endif
