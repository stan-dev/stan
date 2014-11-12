#ifndef STAN__AGRAD__FWD__MATRIX__MDIVIDE_LEFT_LDLT_HPP
#define STAN__AGRAD__FWD__MATRIX__MDIVIDE_LEFT_LDLT_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/matrix/promote_common.hpp>
#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>

namespace stan {
  namespace agrad {

    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */

    template <int R1,int C1,int R2,int C2, typename T2>
    inline Eigen::Matrix<fvar<T2>,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      stan::error_handling::check_multiplicable("mdivide_left_ldlt",
                                                "A", A,
                                                "b", b);

      Eigen::Matrix<T2,R2,C2> b_val(b.rows(), b.cols());
      Eigen::Matrix<T2,R2,C2> b_der(b.rows(), b.cols());
      for (int i = 0; i < b.rows(); i++) 
        for (int j = 0; j < b.cols(); j++) {
          b_val(i,j) = b(i,j).val_;
          b_der(i,j) = b(i,j).d_;
        }

      return to_fvar(mdivide_left_ldlt(A, b_val), 
                     mdivide_left_ldlt(A, b_der));
    }
  }
}
#endif
