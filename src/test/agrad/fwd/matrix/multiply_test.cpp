#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/fvar.hpp>

using stan::agrad::fvar;  
using stan::agrad::multiply;

TEST(AgradFwdMatrix, multiply_scalar_scalar) {
  double d1, d2;
  fvar<double>   v1, v2;

  d1 = 10;
  v1 = 10.0;
  d2 = -2;
  v2 = -2.0;
  
  EXPECT_FLOAT_EQ(-20.0, multiply(d1,d2));
  EXPECT_FLOAT_EQ(-20.0, multiply(d1, v2).val_);
  EXPECT_FLOAT_EQ(-20.0, multiply(v1, d2).val_);
  EXPECT_FLOAT_EQ(-20.0, multiply(v1, v2).val_);

  EXPECT_FLOAT_EQ(6.0, multiply(fvar<double>(3),fvar<double>(2)).val_);
  EXPECT_FLOAT_EQ(6.0, multiply(3.0,fvar<double>(2)).val_);
  EXPECT_FLOAT_EQ(6.0, multiply(fvar<double>(3),2.0).val_);
  EXPECT_FLOAT_EQ(5.0, multiply(fvar<double>(3,1),fvar<double>(2,1)).d_);
  EXPECT_FLOAT_EQ(3.0, multiply(3.0,fvar<double>(2,1)).d_);
  EXPECT_FLOAT_EQ(2.0, multiply(fvar<double>(3,1),2.0).d_);
}

TEST(AgradFwdMatrix, multiply_vector_scalar) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
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

  vector_fv output;
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

TEST(AgradFwdMatrix, multiply_rowvector_scalar) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
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
  
  row_vector_fv output;
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
TEST(AgradFwdMatrix, multiply_matrix_scalar) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  
  matrix_d d1(2,2);
  matrix_fv v1(2,2);
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

  matrix_fv output;
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
TEST(AgradFwdMatrix, multiply_rowvector_vector) {
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  vector_d d2(3);
  vector_fv v2(3);
  
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
TEST(AgradFwdMatrix, multiply_vector_rowvector) {
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  row_vector_d d2(3);
  row_vector_fv v2(3);
  
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

  matrix_fv output = multiply(v1, v2);
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
TEST(AgradFwdMatrix, multiply_matrix_vector) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  matrix_d d1(3,2);
  matrix_fv v1(3,2);
  vector_d d2(2);
  vector_fv v2(2);
  
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

  vector_fv output = multiply(v1, v2);
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
TEST(AgradFwdMatrix, multiply_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::math::vector_d;
  using stan::agrad::vector_fv;

  matrix_d d1(3,2);
  matrix_fv v1(3,2);
  vector_d d2(4);
  vector_fv v2(4);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradFwdMatrix, multiply_rowvector_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);
  
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

  vector_fv output = multiply(v1, v2);
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
TEST(AgradFwdMatrix, multiply_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_fv;

  row_vector_d d1(4);
  row_vector_fv v1(4);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradFwdMatrix, multiply_matrix_matrix) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d1(2,3);
  matrix_fv v1(2,3);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);
  
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

  matrix_fv output = multiply(v1, v2);
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
TEST(AgradFwdMatrix, multiply_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_fv;

  matrix_d d1(2,2);
  matrix_fv v1(2,2);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
