#ifndef STAN__AGRAD__REV__MATRIX__CROSSPROD_HPP
#define STAN__AGRAD__REV__MATRIX__CROSSPROD_HPP

#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/tcrossprod.hpp>

namespace stan {
  namespace agrad {
    
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
