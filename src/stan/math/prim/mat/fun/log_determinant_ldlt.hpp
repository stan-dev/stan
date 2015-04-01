#ifndef STAN__MATH__PRIM__MAT__FUN__LOG_DETERMINANT_LDLT_HPP
#define STAN__MATH__PRIM__MAT__FUN__LOG_DETERMINANT_LDLT_HPP

#include <stan/math/prim/mat/fun/LDLT_factor.hpp>

namespace stan {
  namespace math {

    // Returns log(abs(det(A))) given a LDLT_factor of A
    template<int R, int C, typename T>
    inline T
    log_determinant_ldlt(stan::math::LDLT_factor<T, R, C> &A) {
      return A.log_abs_det();
    }

  }
}
#endif
