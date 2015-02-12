#include <stan/math/prim/mat/fun/diag_pre_multiply.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::diag_pre_multiply;

TEST(MathMatrix,diagPreMultiply) {

  Matrix<double,Dynamic,Dynamic> m(1,1);
  m << 3;

  Matrix<double,Dynamic,1> v(1);
  v << 9;

  Matrix<double,Dynamic,Dynamic> v_m(1,1);
  v_m << 9;
  
  expect_matrix_eq(v_m * m, diag_pre_multiply(v,m));
}
TEST(MathMatrix,diagPreMultiply2) {
  Matrix<double,Dynamic,Dynamic> m(3,3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<double,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  expect_matrix_eq(v_m * m, diag_pre_multiply(v,m));

  Matrix<double,1,Dynamic> rv(3);
  rv << 1, 2, 3;
  expect_matrix_eq(v_m * m, diag_pre_multiply(rv,m));
} 

TEST(MathMatrix,diagPreMultiplyException) {
  Matrix<double,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(diag_pre_multiply(m,m), std::domain_error);

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(diag_pre_multiply(v,m), std::domain_error);
}


