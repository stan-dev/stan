#ifndef __STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_RIGHT_HPP__
#define __STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_RIGHT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>

namespace stan {
  namespace agrad {

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_right(const Eigen::Matrix<fvar<T1>,R1,C1> &b,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &A) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(b,A,"mdivide_left");
      
      return stan::agrad::multiply(b, stan::agrad::inverse(A));
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_right(const Eigen::Matrix<fvar<T1>,R1,C1> &b,
                 const Eigen::Matrix<T2,R2,C2> &A) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(b,A,"mdivide_left");
      
      return stan::agrad::multiply(b,stan::agrad::inverse(to_fvar(A)));
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_right(const Eigen::Matrix<T1,R1,C1> &b,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &A) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(b,A,"mdivide_left");
      
      return stan::agrad::multiply(b,stan::agrad::inverse(A));
    }
  }
}
#endif
