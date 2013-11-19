#ifndef __STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP__
#define __STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>

namespace stan {
  namespace math {

    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      return A.solve(b);
    }
    
  }
}
#endif
