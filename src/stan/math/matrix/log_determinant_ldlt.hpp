#ifndef STAN__MATH__MATRIX__LOG_DETERMINANT_LDLT_HPP
#define STAN__MATH__MATRIX__LOG_DETERMINANT_LDLT_HPP

#include <stan/math/matrix/LDLT_factor.hpp>

namespace stan {
  namespace math {

    // Returns log(abs(det(A))) given a LDLT_factor of A
    template<int R, int C>
    inline double
    log_determinant_ldlt(stan::math::LDLT_factor<double,R,C> &A) {
      return A.log_abs_det();
    }
    
  }
}
#endif
