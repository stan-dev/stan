#ifndef STAN__MATH__MATRIX__CROSSPROD_HPP
#define STAN__MATH__MATRIX__CROSSPROD_HPP

#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/tcrossprod.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of pre-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return Transpose of M times M
     */
    inline matrix_d
    crossprod(const matrix_d& M) {
        return tcrossprod(M.transpose());
    }

  }
}
#endif
