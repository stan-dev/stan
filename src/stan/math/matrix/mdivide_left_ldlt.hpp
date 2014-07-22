#ifndef STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP
#define STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */

    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::check_multiplicable("mdivide_left_ldlt(%1%)",A,"A",
                                      b,"b",(double*)0);
      
      return A.solve(b);
    }
    
  }
}
#endif
