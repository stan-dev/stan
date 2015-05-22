#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>

using stan::math::fvar;  
using stan::math::multiply;

TEST(AgradFwdMatrixOperatorMultiplication,fd_vector_scalar) {
  using stan::math::vector_d;
  using stan::math::vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  double d2;
  fvar<double> v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;  
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  d2 = -2;
  v2 = -2;
  v2.d_ = 1.0;  

  vector_fd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ( 100, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(  -3, output(2).d_);

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ( 100, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(  -3, output(2).d_);

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  -2, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -2, output(2).d_);

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  -2, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -2, output(2).d_);

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  98, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -5, output(2).d_);

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  98, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -5, output(2).d_);
}

TEST(AgradFwdMatrixOperatorMultiplication,fd_rowvector_scalar) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  double d2;
  fvar<double> v2;
  
  d1 << 100, 0, -3;
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;  
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  d2 = -2;
  v2 = -2;
  v2.d_ = 1.0;
  
  row_vector_fd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ( 100, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(  -3, output(2).d_);

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  -2, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -2, output(2).d_);

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  98, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -5, output(2).d_);

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ( 100, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(  -3, output(2).d_);

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  -2, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -2, output(2).d_);

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ(   6, output(2).val_);
  EXPECT_FLOAT_EQ(  98, output(0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1).d_);
  EXPECT_FLOAT_EQ(  -5, output(2).d_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_matrix_scalar) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  
  matrix_d d1(2,2);
  matrix_fd v1(2,2);
  double d2;
  fvar<double> v2;
  
  d1 << 100, 0, -3, 4;
  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0; 
  v1(0,1).d_ = 1.0; 
  v1(1,0).d_ = 1.0; 
  v1(1,1).d_ = 1.0; 
  d2 = -2;
  v2 = -2;
  v2.d_ = 1.0;

  matrix_fd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ( 100, output(0,0).d_);
  EXPECT_FLOAT_EQ(   0, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_);
  EXPECT_FLOAT_EQ(   4, output(1,1).d_);

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_);
 
  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ(  98, output(0,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_);
  EXPECT_FLOAT_EQ(   2, output(1,1).d_);

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ( 100, output(0,0).d_);
  EXPECT_FLOAT_EQ(   0, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_);
  EXPECT_FLOAT_EQ(   4, output(1,1).d_);

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_);
 
  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ(   6, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_);
  EXPECT_FLOAT_EQ(  98, output(0,0).d_);
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_);
  EXPECT_FLOAT_EQ(   2, output(1,1).d_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_rowvector_vector) {
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  vector_d d2(3);
  vector_fd v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  v2(2).d_ = 1.0;

  EXPECT_FLOAT_EQ( 3, multiply(v1, v2).val_);
  EXPECT_FLOAT_EQ( 3, multiply(v1, d2).val_);
  EXPECT_FLOAT_EQ( 3, multiply(d1, v2).val_);
  EXPECT_FLOAT_EQ( 0, multiply(v1, v2).d_);
  EXPECT_FLOAT_EQ( 1, multiply(v1 ,d2).d_);
  EXPECT_FLOAT_EQ(-1, multiply(d1 ,v2).d_);

  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_vector_rowvector) {
  using stan::math::matrix_fd;
  using stan::math::vector_d;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  vector_d d1(3);
  vector_fd v1(3);
  row_vector_d d2(3);
  row_vector_fd v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  v2(2).d_ = 1.0;

  matrix_fd output = multiply(v1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_);
  EXPECT_FLOAT_EQ( -2, output(0,1).val_);
  EXPECT_FLOAT_EQ( -1, output(0,2).val_);
  EXPECT_FLOAT_EQ( 12, output(1,0).val_);
  EXPECT_FLOAT_EQ( -6, output(1,1).val_);
  EXPECT_FLOAT_EQ( -3, output(1,2).val_);
  EXPECT_FLOAT_EQ(-20, output(2,0).val_);  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_);
  EXPECT_FLOAT_EQ(  5, output(2,2).val_);
  EXPECT_FLOAT_EQ(  5, output(0,0).d_);
  EXPECT_FLOAT_EQ( -1, output(0,1).d_);
  EXPECT_FLOAT_EQ(  0, output(0,2).d_);
  EXPECT_FLOAT_EQ(  7, output(1,0).d_);
  EXPECT_FLOAT_EQ(  1, output(1,1).d_);
  EXPECT_FLOAT_EQ(  2, output(1,2).d_);
  EXPECT_FLOAT_EQ( -1, output(2,0).d_);
  EXPECT_FLOAT_EQ( -7, output(2,1).d_);
  EXPECT_FLOAT_EQ( -6, output(2,2).d_);
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_);
  EXPECT_FLOAT_EQ( -2, output(0,1).val_);
  EXPECT_FLOAT_EQ( -1, output(0,2).val_);
  EXPECT_FLOAT_EQ( 12, output(1,0).val_);
  EXPECT_FLOAT_EQ( -6, output(1,1).val_);
  EXPECT_FLOAT_EQ( -3, output(1,2).val_);
  EXPECT_FLOAT_EQ(-20, output(2,0).val_);  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_);
  EXPECT_FLOAT_EQ(  5, output(2,2).val_);
  EXPECT_FLOAT_EQ(  4, output(0,0).d_);
  EXPECT_FLOAT_EQ( -2, output(0,1).d_);
  EXPECT_FLOAT_EQ( -1, output(0,2).d_);
  EXPECT_FLOAT_EQ(  4, output(1,0).d_);
  EXPECT_FLOAT_EQ( -2, output(1,1).d_);
  EXPECT_FLOAT_EQ( -1, output(1,2).d_);
  EXPECT_FLOAT_EQ(  4, output(2,0).d_);
  EXPECT_FLOAT_EQ( -2, output(2,1).d_);
  EXPECT_FLOAT_EQ( -1, output(2,2).d_);
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_);
  EXPECT_FLOAT_EQ( -2, output(0,1).val_);
  EXPECT_FLOAT_EQ( -1, output(0,2).val_);
  EXPECT_FLOAT_EQ( 12, output(1,0).val_);
  EXPECT_FLOAT_EQ( -6, output(1,1).val_);
  EXPECT_FLOAT_EQ( -3, output(1,2).val_);
  EXPECT_FLOAT_EQ(-20, output(2,0).val_);  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_);
  EXPECT_FLOAT_EQ(  5, output(2,2).val_);
  EXPECT_FLOAT_EQ(  1, output(0,0).d_);
  EXPECT_FLOAT_EQ(  1, output(0,1).d_);
  EXPECT_FLOAT_EQ(  1, output(0,2).d_);
  EXPECT_FLOAT_EQ(  3, output(1,0).d_);
  EXPECT_FLOAT_EQ(  3, output(1,1).d_);
  EXPECT_FLOAT_EQ(  3, output(1,2).d_);
  EXPECT_FLOAT_EQ( -5, output(2,0).d_);
  EXPECT_FLOAT_EQ( -5, output(2,1).d_);
  EXPECT_FLOAT_EQ( -5, output(2,2).d_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_matrix_vector) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  matrix_d d1(3,2);
  matrix_fd v1(3,2);
  vector_d d2(2);
  vector_fd v2(2);
  
  d1 << 1, 3, -5, 4, -2, -1;
  v1 << 1, 3, -5, 4, -2, -1;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v1(2,0).d_ = 1.0;
  v1(2,1).d_ = 1.0;
  d2 << -2, 4;
  v2 << -2, 4;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;

  vector_fd output = multiply(v1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_);
  EXPECT_FLOAT_EQ(26, output(1).val_);
  EXPECT_FLOAT_EQ( 0, output(2).val_);
  EXPECT_FLOAT_EQ( 6, output(0).d_);
  EXPECT_FLOAT_EQ( 1, output(1).d_);
  EXPECT_FLOAT_EQ(-1, output(2).d_);
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_);
  EXPECT_FLOAT_EQ(26, output(1).val_);
  EXPECT_FLOAT_EQ( 0, output(2).val_);
  EXPECT_FLOAT_EQ( 2, output(0).d_);
  EXPECT_FLOAT_EQ( 2, output(1).d_);
  EXPECT_FLOAT_EQ( 2, output(2).d_);
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_);
  EXPECT_FLOAT_EQ(26, output(1).val_);
  EXPECT_FLOAT_EQ( 0, output(2).val_);
  EXPECT_FLOAT_EQ( 4, output(0).d_);
  EXPECT_FLOAT_EQ(-1, output(1).d_);
  EXPECT_FLOAT_EQ(-3, output(2).d_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_d;
  using stan::math::vector_fd;

  matrix_d d1(3,2);
  matrix_fd v1(3,2);
  vector_d d2(4);
  vector_fd v2(4);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_rowvector_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::vector_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(3);
  row_vector_fd v1(3);
  matrix_d d2(3,2);
  matrix_fd v2(3,2);
  
  d1 << -2, 4, 1;
  v1 << -2, 4, 1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;
  v2(0,0).d_ = 1.0;
  v2(0,1).d_ = 1.0;
  v2(1,0).d_ = 1.0;
  v2(1,1).d_ = 1.0;
  v2(2,0).d_ = 1.0;
  v2(2,1).d_ = 1.0;

  vector_fd output = multiply(v1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_);
  EXPECT_FLOAT_EQ(  9, output(1).val_);
  EXPECT_FLOAT_EQ( -3, output(0).d_);
  EXPECT_FLOAT_EQ(  9, output(1).d_);

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_);
  EXPECT_FLOAT_EQ(  9, output(1).val_);
  EXPECT_FLOAT_EQ( -6, output(0).d_);
  EXPECT_FLOAT_EQ(  6, output(1).d_);
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_);
  EXPECT_FLOAT_EQ(  9, output(1).val_);
  EXPECT_FLOAT_EQ(  3, output(0).d_);
  EXPECT_FLOAT_EQ(  3, output(1).d_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fd;

  row_vector_d d1(4);
  row_vector_fd v1(4);
  matrix_d d2(3,2);
  matrix_fd v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d d1(2,3);
  matrix_fd v1(2,3);
  matrix_d d2(3,2);
  matrix_fd v2(3,2);
  
  d1 << 9, 24, 3, 46, -9, -33;
  v1 << 9, 24, 3, 46, -9, -33;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(0,2).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v1(1,2).d_ = 1.0;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;
  v2(0,0).d_ = 1.0;
  v2(0,1).d_ = 1.0;
  v2(1,0).d_ = 1.0;
  v2(1,1).d_ = 1.0;
  v2(2,0).d_ = 1.0;
  v2(2,1).d_ = 1.0;

  matrix_fd output = multiply(v1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_);
  EXPECT_FLOAT_EQ( 120, output(0,1).val_);
  EXPECT_FLOAT_EQ( 157, output(1,0).val_);
  EXPECT_FLOAT_EQ( 135, output(1,1).val_);
  EXPECT_FLOAT_EQ(  30, output(0,0).d_);
  EXPECT_FLOAT_EQ(  42, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_);
  EXPECT_FLOAT_EQ(  10, output(1,1).d_);

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_);
  EXPECT_FLOAT_EQ( 120, output(0,1).val_);
  EXPECT_FLOAT_EQ( 157, output(1,0).val_);
  EXPECT_FLOAT_EQ( 135, output(1,1).val_);
  EXPECT_FLOAT_EQ(  -6, output(0,0).d_);
  EXPECT_FLOAT_EQ(   6, output(0,1).d_);
  EXPECT_FLOAT_EQ(  -6, output(1,0).d_);
  EXPECT_FLOAT_EQ(   6, output(1,1).d_);
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_);
  EXPECT_FLOAT_EQ( 120, output(0,1).val_);
  EXPECT_FLOAT_EQ( 157, output(1,0).val_);
  EXPECT_FLOAT_EQ( 135, output(1,1).val_);
}
TEST(AgradFwdMatrixOperatorMultiplication,fd_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fd;

  matrix_d d1(2,2);
  matrix_fd v1(2,2);
  matrix_d d2(3,2);
  matrix_fd v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_vector_scalar) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1(3);
  vector_ffd v1(3);
  double d2;
  fvar<fvar<double> > v2;
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  d2 = -2;
  v2.val_.val_ = -2;
  v2.d_.val_ = 1.0;  

  vector_ffd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val());

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val());
}

TEST(AgradFwdMatrixOperatorMultiplication,ffd_rowvector_scalar) {
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  double d2;
  fvar<fvar<double> > v2;
  
  fvar<fvar<double> > a,b,c;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;

  d1 << 100, 0, -3;
  v1 << a,b,c;
  d2 = -2;
  v2.val_.val_ = -2;
  v2.d_.val_ = 1.0;
  
  row_vector_ffd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val());

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_matrix_scalar) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  
  matrix_d d1(2,2);
  matrix_ffd v1(2,2);
  double d2;
  fvar<fvar<double> > v2;

  fvar<fvar<double> > a,b,c,d,e;
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  d.val_.val_ = 4.0;
  e.val_.val_ = -2.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;

  d1 << 100, 0, -3, 4;
  v1 << a,b,c,d;
  d2 = -2;
  v2 = e;

  matrix_ffd output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(   4, output(1,1).d_.val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_.val());
 
  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(   2, output(1,1).d_.val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 100, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(   4, output(1,1).d_.val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_.val());
 
  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  98, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(   2, output(1,1).d_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_rowvector_vector) {
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  vector_d d2(3);
  vector_ffd v2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = -2.0;
  b.val_.val_ = 4.0;
  c.val_.val_ = 1.0;
  d.val_.val_ = 3.0;
  e.val_.val_ = -5.0;
  f.val_.val_ = -1.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << c,d,e;
  d2 << 4, -2, -1;
  v2 << b,a,f;

  EXPECT_FLOAT_EQ( 3, multiply(v1, v2).val_.val());
  EXPECT_FLOAT_EQ( 3, multiply(v1, d2).val_.val());
  EXPECT_FLOAT_EQ( 3, multiply(d1, v2).val_.val());
  EXPECT_FLOAT_EQ( 0, multiply(v1, v2).d_.val());
  EXPECT_FLOAT_EQ( 1, multiply(v1 ,d2).d_.val());
  EXPECT_FLOAT_EQ(-1, multiply(d1 ,v2).d_.val());

  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_vector_rowvector) {
  using stan::math::matrix_ffd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  vector_d d1(3);
  vector_ffd v1(3);
  row_vector_d d2(3);
  row_vector_ffd v2(3);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = -2.0;
  b.val_.val_ = 4.0;
  c.val_.val_ = 1.0;
  d.val_.val_ = 3.0;
  e.val_.val_ = -5.0;
  f.val_.val_ = -1.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5;
  v1 << c,d,e;
  d2 << 4, -2, -1;
  v2 << b,a,f;

  matrix_ffd output = multiply(v1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val());
  EXPECT_FLOAT_EQ(  5, output(0,0).d_.val());
  EXPECT_FLOAT_EQ( -1, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  0, output(0,2).d_.val());
  EXPECT_FLOAT_EQ(  7, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  1, output(1,1).d_.val());
  EXPECT_FLOAT_EQ(  2, output(1,2).d_.val());
  EXPECT_FLOAT_EQ( -1, output(2,0).d_.val());
  EXPECT_FLOAT_EQ( -7, output(2,1).d_.val());
  EXPECT_FLOAT_EQ( -6, output(2,2).d_.val());
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val());
  EXPECT_FLOAT_EQ(  4, output(0,0).d_.val());
  EXPECT_FLOAT_EQ( -2, output(0,1).d_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).d_.val());
  EXPECT_FLOAT_EQ(  4, output(1,0).d_.val());
  EXPECT_FLOAT_EQ( -2, output(1,1).d_.val());
  EXPECT_FLOAT_EQ( -1, output(1,2).d_.val());
  EXPECT_FLOAT_EQ(  4, output(2,0).d_.val());
  EXPECT_FLOAT_EQ( -2, output(2,1).d_.val());
  EXPECT_FLOAT_EQ( -1, output(2,2).d_.val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val());
  EXPECT_FLOAT_EQ(  1, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  1, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  1, output(0,2).d_.val());
  EXPECT_FLOAT_EQ(  3, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  3, output(1,1).d_.val());
  EXPECT_FLOAT_EQ(  3, output(1,2).d_.val());
  EXPECT_FLOAT_EQ( -5, output(2,0).d_.val());
  EXPECT_FLOAT_EQ( -5, output(2,1).d_.val());
  EXPECT_FLOAT_EQ( -5, output(2,2).d_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_matrix_vector) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  matrix_d d1(3,2);
  matrix_ffd v1(3,2);
  vector_d d2(2);
  vector_ffd v2(2);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = -2.0;
  b.val_.val_ = 4.0;
  c.val_.val_ = 1.0;
  d.val_.val_ = 3.0;
  e.val_.val_ = -5.0;
  f.val_.val_ = -1.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;

  d1 << 1, 3, -5, 4, -2, -1;
  v1 << c,d,e,b,a,f;
  d2 << -2, 4;
  v2 << a,b;

  vector_ffd output = multiply(v1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val());
  EXPECT_FLOAT_EQ( 6, output(0).d_.val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val());
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val());
  EXPECT_FLOAT_EQ( 2, output(0).d_.val());
  EXPECT_FLOAT_EQ( 2, output(1).d_.val());
  EXPECT_FLOAT_EQ( 2, output(2).d_.val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val());
  EXPECT_FLOAT_EQ(-3, output(2).d_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  matrix_d d1(3,2);
  matrix_ffd v1(3,2);
  vector_d d2(4);
  vector_ffd v2(4);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_rowvector_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  matrix_d d2(3,2);
  matrix_ffd v2(3,2);
  
  fvar<fvar<double> > a,b,c,d,e,f;
  a.val_.val_ = -2.0;
  b.val_.val_ = 4.0;
  c.val_.val_ = 1.0;
  d.val_.val_ = 3.0;
  e.val_.val_ = -5.0;
  f.val_.val_ = -1.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;

  d1 << -2, 4, 1;
  v1 << a,b,c;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << c,d,e,b,a,f;

  vector_ffd output = multiply(v1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val());
  EXPECT_FLOAT_EQ( -3, output(0).d_.val());
  EXPECT_FLOAT_EQ(  9, output(1).d_.val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val());
  EXPECT_FLOAT_EQ( -6, output(0).d_.val());
  EXPECT_FLOAT_EQ(  6, output(1).d_.val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val());
  EXPECT_FLOAT_EQ(  3, output(0).d_.val());
  EXPECT_FLOAT_EQ(  3, output(1).d_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(4);
  row_vector_ffd v1(4);
  matrix_d d2(3,2);
  matrix_ffd v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d1(2,3);
  matrix_ffd v1(2,3);
  matrix_d d2(3,2);
  matrix_ffd v2(3,2);
  
  fvar<fvar<double> > a,b,c,d,e,f,g,h,i,j,k,l;
  a.val_.val_ = 9.0;
  b.val_.val_ = 24.0;
  c.val_.val_ = 3.0;
  d.val_.val_ = 46.0;
  e.val_.val_ = -9.0;
  f.val_.val_ = -33.0;
  g.val_.val_ = 1.0;
  h.val_.val_ = 3.0;
  i.val_.val_ = -5.0;
  j.val_.val_ = 4.0;
  k.val_.val_ = -2.0;
  l.val_.val_ = -1.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  e.d_.val_ = 1.0;
  f.d_.val_ = 1.0;
  g.d_.val_ = 1.0;
  h.d_.val_ = 1.0;
  i.d_.val_ = 1.0;
  j.d_.val_ = 1.0;
  k.d_.val_ = 1.0;
  l.d_.val_ = 1.0;

  d1 << 9, 24, 3, 46, -9, -33;
  v1 << a,b,c,d,e,f;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << g,h,i,j,k,l;

  matrix_ffd output = multiply(v1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  30, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  42, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  10, output(1,1).d_.val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(  -6, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(   6, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  -6, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(   6, output(1,1).d_.val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val());
}
TEST(AgradFwdMatrixOperatorMultiplication,ffd_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d1(2,2);
  matrix_ffd v1(2,2);
  matrix_d d2(3,2);
  matrix_ffd v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
