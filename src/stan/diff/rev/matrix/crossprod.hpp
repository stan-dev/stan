#ifndef __STAN__DIFF__REV__MATRIX__CROSSPROD_HPP__
#define __STAN__DIFF__REV__MATRIX__CROSSPROD_HPP__

#include <stan/diff/rev/matrix/typedefs.hpp>
#include <stan/diff/rev/matrix/tcrossprod.hpp>

namespace stan {
  namespace diff {
    
    /**
     * Returns the result of pre-multiplying a matrix by its
     * own transpose.
     * @param M Matrix to multiply.
     * @return Transpose of M times M
     */
    inline matrix_v
    crossprod(const matrix_v& M) {
      return tcrossprod(M.transpose());
    }

  }
}
#endif
