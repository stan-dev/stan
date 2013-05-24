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
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>

namespace stan {
  namespace agrad {

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      using stan::agrad::multiply;
      using stan::math::multiply;      
      using stan::agrad::inverse;
      using stan::math::inverse;
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");

      Eigen::Matrix<fvar<T1>,R1,C1> fvar_inv_A;
      fvar_inv_A = stan::agrad::inverse(A);


      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> inv_A_mult_b;
      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> val;
      Eigen::Matrix<T1,R1,C1> val_A; 
      Eigen::Matrix<T1,R1,C2> inv_A; 
      Eigen::Matrix<T1,R1,C1> deriv_A; 
      Eigen::Matrix<T2,R2,C2> val_b; 
      Eigen::Matrix<T2,R2,C2> deriv_b; 

      for (int i = 0; i < A.rows(); i++) {
        for(int j = 0; j < A.cols(); j++) {
          inv_A(i,j) = fvar_inv_A(i,j).val_;
          val_A(i,j) = A(i,j).d_;
          deriv_A(i,j) = A(i,j).d_;
          val_b(i,j) = b(i,j).val_;
          deriv_b(i,j) = b(i,j).d_;
        }
      }

      inv_A_mult_b = multiply(inverse(val_A), val_b);

      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> deriv;
      deriv =  multiply(inv_A, val_b);
      deriv = multiply(deriv_A, deriv);
      deriv = multiply(inv_A, deriv_b - deriv);

      return stan::agrad::to_fvar(inv_A_mult_b, deriv);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<fvar<T1>,R1,C1> &A,
                 const Eigen::Matrix<T2,R2,C2> &b) {
      
      using stan::agrad::multiply;
      using stan::math::multiply;      
      using stan::agrad::inverse;
      using stan::math::inverse;
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");

      Eigen::Matrix<fvar<T1>,R1,C1> fvar_inv_A;
      fvar_inv_A = stan::agrad::inverse(A);

      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> inv_A_mult_b;
      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> val;
      Eigen::Matrix<T1,R1,C1> val_A; 
      Eigen::Matrix<T1,R1,C1> deriv_A;
      Eigen::Matrix<T1,R1,C1> inv_A; 

      for (int i = 0; i < A.rows(); i++) {
        for(int j = 0; j < A.cols(); j++) {
          inv_A(i,j) = fvar_inv_A(i,j).val_;
          val_A(i,j) = A(i,j).d_;
          deriv_A(i,j) = A(i,j).d_;
        }
      }

      inv_A_mult_b = multiply(inverse(val_A), b);

      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> deriv;
      deriv = multiply(inv_A, b);
      deriv = multiply(deriv_A, deriv);
      deriv = -multiply(inv_A, deriv);

      return stan::agrad::to_fvar(inv_A_mult_b, deriv);
    }

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C2>
    mdivide_left(const Eigen::Matrix<T1,R1,C1> &A,
                 const Eigen::Matrix<fvar<T2>,R2,C2> &b) {
      
      using stan::agrad::multiply;
      using stan::math::multiply;      
      using stan::agrad::inverse;
      using stan::math::inverse;
      stan::math::validate_square(A,"mdivide_left");
      stan::math::validate_multiplicable(A,b,"mdivide_left");

      Eigen::Matrix<T1,R1,C1> inv_A;
      inv_A = inverse(A);

      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> inv_A_mult_b;
      Eigen::Matrix<T2,R2,C2> deriv_b; 
      Eigen::Matrix<T2,R2,C2> val_b; 

      for (int i = 0; i < A.rows(); i++) {
        for(int j = 0; j < A.cols(); j++) {
          deriv_b(i,j) = b(i,j).d_;
          val_b(i,j) = b(i,j).val_;
        }
      }

      inv_A_mult_b = multiply(inv_A, val_b);

      Eigen::Matrix<typename stan::return_type<T1,T2>,R1,C2> deriv;
      deriv = multiply(inv_A, deriv_b);
      return stan::agrad::to_fvar(inv_A_mult_b, deriv);
    }
  }
}
#endif
