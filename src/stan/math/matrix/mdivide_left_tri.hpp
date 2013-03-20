#ifndef __STAN__MATH__MATRIX__MDIVIDE_LEFT_TRI_HPP__
#define __STAN__MATH__MATRIX__MDIVIDE_LEFT_TRI_HPP__

#include <stan/math/matrix.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the solution of the system Ax=b when A is triangular and b=I.
     * @param A Triangular matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @return x = A^-1 .
     * @throws std::domain_error if A is not square
     */
    template<int TriView, typename T,int R1, int C1>
    inline 
    Eigen::Matrix<T,R1,C1> 
    mdivide_left_tri(const Eigen::Matrix<T,R1,C1> &A) {
      stan::math::validate_square(A,"mdivide_left_tri");
      int n = A.rows();
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> b;
      b.setIdentity(n,n);
      A.template triangularView<TriView>().solveInPlace(b);
      return b;
    }

  }
}
#endif
