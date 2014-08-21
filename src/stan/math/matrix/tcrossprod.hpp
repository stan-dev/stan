#ifndef STAN__MATH__MATRIX__TCROSSPROD_HPP
#define STAN__MATH__MATRIX__TCROSSPROD_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of post-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return M times its transpose.
     */
    inline matrix_d
    tcrossprod(const matrix_d& M) {
        if (M.rows() == 0)
          return matrix_d(0,0);
        if (M.rows() == 1)
          return M * M.transpose();
        matrix_d result(M.rows(),M.rows());
        return result
          .setZero()
          .selfadjointView<Eigen::Upper>()
          .rankUpdate(M);
    }

  }
}
#endif
