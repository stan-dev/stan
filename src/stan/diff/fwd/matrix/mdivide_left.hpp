#ifndef __STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__
#define __STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_LEFT_HPP__

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
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::agrad::multiply(stan::agrad::inverse(A), b);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<T2,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::agrad::multiply(stan::agrad::inverse(A), b);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<T1,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");
      
      return stan::agrad::multiply(stan::agrad::inverse(to_fvar(A)), b);
    }
  }
}
#endif
