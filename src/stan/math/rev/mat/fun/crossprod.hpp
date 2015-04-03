#ifndef STAN_MATH_REV_MAT_FUN_CROSSPROD_HPP
#define STAN_MATH_REV_MAT_FUN_CROSSPROD_HPP

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
