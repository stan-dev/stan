#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/diag_pre_multiply.hpp>

TEST(AgradFwdMatrix, diag_pre_multiply_vector) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fv Y(3,3);
  Y << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  vector_d A(3);
  A << 1, 2, 3;
  vector_fv B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fv output = stan::math::diag_pre_multiply(B,Y);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_);
  EXPECT_FLOAT_EQ( 6,output(0,1).d_);
  EXPECT_FLOAT_EQ( 8,output(0,2).d_);
  EXPECT_FLOAT_EQ( 8,output(1,0).d_);
  EXPECT_FLOAT_EQ(10,output(1,1).d_);
  EXPECT_FLOAT_EQ(12,output(1,2).d_);
  EXPECT_FLOAT_EQ(14,output(2,0).d_);
  EXPECT_FLOAT_EQ(16,output(2,1).d_);
  EXPECT_FLOAT_EQ(18,output(2,2).d_);
}
TEST(AgradFwdMatrix, diag_pre_multiply_vector_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;


  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  vector_fv B(3);
  vector_fv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrix, diag_pre_multiply_rowvector) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fv Y(3,3);
  Y << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
   Y(2,2).d_ = 2.0;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_fv B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fv output = stan::math::diag_pre_multiply(B,Y);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_);
  EXPECT_FLOAT_EQ( 6,output(0,1).d_);
  EXPECT_FLOAT_EQ( 8,output(0,2).d_);
  EXPECT_FLOAT_EQ( 8,output(1,0).d_);
  EXPECT_FLOAT_EQ(10,output(1,1).d_);
  EXPECT_FLOAT_EQ(12,output(1,2).d_);
  EXPECT_FLOAT_EQ(14,output(2,0).d_);
  EXPECT_FLOAT_EQ(16,output(2,1).d_);
  EXPECT_FLOAT_EQ(18,output(2,2).d_);
}
TEST(AgradFwdMatrix, diag_pre_multiply_rowvector_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;


  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  row_vector_fv B(3);
  row_vector_fv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
