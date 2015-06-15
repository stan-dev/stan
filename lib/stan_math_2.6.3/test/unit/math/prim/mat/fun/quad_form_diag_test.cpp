#include <stan/math/prim/mat/fun/quad_form_diag.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::quad_form_diag;

TEST(MathMatrix,quadFormDiag) {

  Matrix<double,Dynamic,Dynamic> m(1,1);
  m << 3;

  Matrix<double,Dynamic,1> v(1);
  v << 9;

  Matrix<double,Dynamic,Dynamic> v_m(1,1);
  v_m << 9;
  
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,v));
}
TEST(MathMatrix,quadFormDiag2) {
  Matrix<double,Dynamic,Dynamic> m(3,3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<double,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,v));

  Matrix<double,1,Dynamic> rv(3);
  rv << 1, 2, 3;
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,rv));
}

TEST(MathMatrix,quadFormDiagException) {
  Matrix<double,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(quad_form_diag(m,m), std::invalid_argument);

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(quad_form_diag(m,v), std::domain_error);
  
  Matrix<double,Dynamic,Dynamic> m2(3,2);
  m2 << 
    2, 3,
    4, 5,
    6, 7;
    
  Matrix<double,Dynamic,1> v2(2);
  v2 << 1, 2;

  EXPECT_THROW(quad_form_diag(m2,v), std::invalid_argument);
  EXPECT_THROW(quad_form_diag(m2,v2), std::invalid_argument);
}


