#ifndef __STAN__DIFF__FWD__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__
#define __STAN__DIFF__FWD__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/inverse.hpp>
#include <stan/diff/fwd/matrix/multiply.hpp>

namespace stan {
  namespace diff {

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::diff::multiply(stan::diff::inverse(A), b);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<T2,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::diff::multiply(stan::diff::inverse(A), b);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<T1,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::diff::multiply(stan::diff::inverse(to_fvar(A)), b);
    }
  }
}
#endif
