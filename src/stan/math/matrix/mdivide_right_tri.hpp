#ifndef __STAN__MATH__MATRIX__MDIVIDE_RIGHT_TRI_HPP__
#define __STAN__MATH__MATRIX__MDIVIDE_RIGHT_TRI_HPP__

#include <stdexcept>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the solution of the system Ax=b when A is triangular
     * @param A Triangular matrix.  Specify upper or lower with TriView
     * being Eigen::Upper or Eigen::Lower.
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if A is not square or the rows of b don't
     * match the size of A.
     */
    template <int TriView, typename T1, typename T2, 
              int R1, int C1, int R2, int C2>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_right_tri(const Eigen::Matrix<T1,R1,C1> &b,
                      const Eigen::Matrix<T2,R2,C2> &A) {
      stan::math::validate_square(A,"mdivide_left_tri_low");
      stan::math::validate_multiplicable(b,A,"mdivide_right_tri");
      // FIXME: This is nice and general but requires some extra memory and copying.
      if (TriView == Eigen::Lower) {
        return transpose(mdivide_left_tri<Eigen::Upper>(transpose(A),transpose(b)));
      }
      else if (TriView == Eigen::Upper) {
        return transpose(mdivide_left_tri<Eigen::Lower>(transpose(A),transpose(b)));
      }
      else {
        throw std::domain_error("triangular view must be Eigen::Lower or Eigen::Upper");
      }
    }

  }
}
#endif
