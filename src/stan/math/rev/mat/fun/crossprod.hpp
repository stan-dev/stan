#ifndef STAN__MATH__REV__MAT__FUN__CROSSPROD_HPP
#define STAN__MATH__REV__MAT__FUN__CROSSPROD_HPP

#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/mat/fun/tcrossprod.hpp>

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
