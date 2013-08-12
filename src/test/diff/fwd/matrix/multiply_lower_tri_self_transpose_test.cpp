#include <stan/agrad/fwd/matrix/multiply_lower_tri_self_transpose.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>

TEST(AgradFwdMatrix, multiply_lower_tri_self_transpose_3x3_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  matrix_fv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
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
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fv output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_);
  EXPECT_FLOAT_EQ( 6,output(0,1).d_);
  EXPECT_FLOAT_EQ(10,output(0,2).d_);
  EXPECT_FLOAT_EQ( 6,output(1,0).d_);
  EXPECT_FLOAT_EQ(20,output(1,1).d_);
  EXPECT_FLOAT_EQ(28,output(1,2).d_);
  EXPECT_FLOAT_EQ(10,output(2,0).d_);
  EXPECT_FLOAT_EQ(28,output(2,1).d_);
  EXPECT_FLOAT_EQ(60,output(2,2).d_);
}
TEST(AgradFwdMatrix, multiply_lower_tri_self_transpose_3x2_matrix) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(3,2);
  Z << 1, 0, 0,
    2, 3, 0;
  matrix_fv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fv output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_);
  EXPECT_FLOAT_EQ( 2,output(0,1).d_);
  EXPECT_FLOAT_EQ( 8,output(0,2).d_);
  EXPECT_FLOAT_EQ( 2,output(1,0).d_);
  EXPECT_FLOAT_EQ( 8,output(1,1).d_);
  EXPECT_FLOAT_EQ(10,output(1,2).d_);
  EXPECT_FLOAT_EQ( 8,output(2,0).d_);
  EXPECT_FLOAT_EQ(10,output(2,1).d_);
  EXPECT_FLOAT_EQ(12,output(2,2).d_);
}
