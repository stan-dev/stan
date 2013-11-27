#ifndef __STAN__MATH__MATRIX__MDIVIDE_RIGHT_LDLT_HPP__
#define __STAN__MATH__MATRIX__MDIVIDE_RIGHT_LDLT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_right_ldlt(const Eigen::Matrix<T1,R1,C1> &b,
                       const stan::math::LDLT_factor<T2,R2,C2> &A) {
      stan::math::validate_multiplicable(b,A,"mdivide_right_ldlt");

      return transpose(mdivide_left_ldlt(A,transpose(b)));
    }
    
    template <int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<double,R1,C2>
    mdivide_right_ldlt(const Eigen::Matrix<double,R1,C1> &b,
                       const stan::math::LDLT_factor<double,R2,C2> &A) {
      stan::math::validate_multiplicable(b,A,"mdivide_right_ldlt");

      return A.solveRight(b);
    }

  }
}
#endif
