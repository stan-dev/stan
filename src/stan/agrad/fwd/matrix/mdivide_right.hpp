#ifndef STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_RIGHT_HPP
#define STAN__AGRAD__FWD__MATRIX__COLUMNS_MDIVIDE_RIGHT_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/mdivide_right.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/math/matrix/mdivide_right.hpp>

namespace stan {
  namespace agrad {

    template <typename T, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<T>,R1,C2>
    mdivide_right(const Eigen::Matrix<fvar<T>,R1,C1> &A,
                  const Eigen::Matrix<fvar<T>,R2,C2> &b) {
      
      using stan::math::multiply;      
      using stan::math::mdivide_right;
      stan::error_handling::check_square("mdivide_right", "b", b);
      stan::error_handling::check_multiplicable("mdivide_right",
                                                "A", A,
                                                "b", b);

      Eigen::Matrix<T,R1,C2> A_mult_inv_b(A.rows(),b.cols());
      Eigen::Matrix<T,R1,C2> deriv_A_mult_inv_b(A.rows(),b.cols());
      Eigen::Matrix<T,R2,C2> deriv_b_mult_inv_b(b.rows(),b.cols());
      Eigen::Matrix<T,R1,C1> val_A(A.rows(),A.cols()); 
      Eigen::Matrix<T,R1,C1> deriv_A(A.rows(),A.cols()); 
      Eigen::Matrix<T,R2,C2> val_b(b.rows(),b.cols()); 
      Eigen::Matrix<T,R2,C2> deriv_b(b.rows(),b.cols()); 

      for (int j = 0; j < A.cols(); j++) {
        for(int i = 0; i < A.rows(); i++) {
          val_A(i,j) = A(i,j).val_;
          deriv_A(i,j) = A(i,j).d_;
        }
      }

      for (int j = 0; j < b.cols(); j++) {
        for(int i = 0; i < b.rows(); i++) {
          val_b(i,j) = b(i,j).val_;
          deriv_b(i,j) = b(i,j).d_;
        }
      }

      A_mult_inv_b = mdivide_right(val_A, val_b);
      deriv_A_mult_inv_b = mdivide_right(deriv_A, val_b);
      deriv_b_mult_inv_b = mdivide_right(deriv_b, val_b);

      Eigen::Matrix<T,R1,C2> deriv(A.rows(), b.cols());
      deriv = deriv_A_mult_inv_b - multiply(A_mult_inv_b, deriv_b_mult_inv_b);

      return stan::agrad::to_fvar(A_mult_inv_b, deriv);
    }

    template <typename T, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<T>,R1,C2>
    mdivide_right(const Eigen::Matrix<fvar<T>,R1,C1> &A,
                  const Eigen::Matrix<double,R2,C2> &b) {
      
      using stan::math::multiply;      
      using stan::math::mdivide_right;
      stan::error_handling::check_square("mdivide_right", "b", b);
      stan::error_handling::check_multiplicable("mdivide_right",
                                                "A", A,
                                                "b", b);

      Eigen::Matrix<T,R2,C2> deriv_b_mult_inv_b(b.rows(),b.cols());
      Eigen::Matrix<T,R1,C1> val_A(A.rows(),A.cols()); 
      Eigen::Matrix<T,R1,C1> deriv_A(A.rows(),A.cols());

      for (int j = 0; j < A.cols(); j++) {
        for(int i = 0; i < A.rows(); i++) {
          val_A(i,j) = A(i,j).val_;
          deriv_A(i,j) = A(i,j).d_;
        }
      }

      return stan::agrad::to_fvar(mdivide_right(val_A, b), 
                                  mdivide_right(deriv_A, b));
    }

    template <typename T, int R1,int C1,int R2,int C2>
    inline 
    Eigen::Matrix<fvar<T>,R1,C2>
    mdivide_right(const Eigen::Matrix<double,R1,C1> &A,
                  const Eigen::Matrix<fvar<T>,R2,C2> &b) {
      
      using stan::math::multiply;      
      using stan::math::mdivide_right;
      stan::error_handling::check_square("mdivide_right", "b", b);
      stan::error_handling::check_multiplicable("mdivide_right",
                                                "A", A,
                                                "b", b);
      Eigen::Matrix<T,R1,C2> 
        A_mult_inv_b(A.rows(),b.cols());
      Eigen::Matrix<T,R2,C2> deriv_b_mult_inv_b(b.rows(),b.cols());
      Eigen::Matrix<T,R2,C2> val_b(b.rows(),b.cols()); 
      Eigen::Matrix<T,R2,C2> deriv_b(b.rows(),b.cols()); 

      for (int j = 0; j < b.cols(); j++) {
        for(int i = 0; i < b.rows(); i++) {
          val_b(i,j) = b(i,j).val_;
          deriv_b(i,j) = b(i,j).d_;
        }
      }

      A_mult_inv_b = mdivide_right(A, val_b);
      deriv_b_mult_inv_b = mdivide_right(deriv_b, val_b);

      Eigen::Matrix<T,R1,C2> 
        deriv(A.rows(), b.cols());
      deriv = -multiply(A_mult_inv_b, deriv_b_mult_inv_b);

      return stan::agrad::to_fvar(A_mult_inv_b, deriv);
    }
  }
}
#endif
