#include <stan/agrad/fwd/matrix/divide.hpp>
#include <gtest/gtest.h>
#include<stan/agrad/fwd/fvar.hpp>
#include <stan/math/matrix/divide.hpp>

using stan::agrad::fvar;

TEST(AgradFwdMatrix, divide_scalar) {
  using stan::agrad::divide;
  double d1, d2;
  fvar<double>   v1, v2;

  d1 = 10;
  v1 = 10;
   v1.d_ = 1.0;
  d2 = -2;
  v2 = -2;
   v2.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(  -5, divide(d1, d2));
  EXPECT_FLOAT_EQ(  -5, divide(d1, v2).val_);
  EXPECT_FLOAT_EQ(  -5, divide(v1, d2).val_);
  EXPECT_FLOAT_EQ(  -5, divide(v1, v2).val_);
  EXPECT_FLOAT_EQ(-2.5, divide(d1, v2).d_);
  EXPECT_FLOAT_EQ(-0.5, divide(v1, d2).d_);
  EXPECT_FLOAT_EQ(  -3, divide(v1, v2).d_);

  d2 = 0;
  v2 = 0;

  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(d1, d2));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(d1, v2).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(v1, d2).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(v1, v2).val_);
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_));
  EXPECT_TRUE(std::isnan(divide(v1, d2).d_));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_));

  d1 = 0;
  v1 = 0;
  EXPECT_TRUE(std::isnan(divide(d1, d2)));
  EXPECT_TRUE(std::isnan(divide(d1, v2).val_));
  EXPECT_TRUE(std::isnan(divide(v1, d2).val_));
  EXPECT_TRUE(std::isnan(divide(v1, v2).val_));
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_));
  EXPECT_TRUE(std::isnan(divide(v1, d2).d_));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_));
}
TEST(AgradFwdMatrix, divide_vector) {
  using stan::math::divide;
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
  
  vector_d output_d;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  vector_fv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(2).val_);
  EXPECT_FLOAT_EQ( -25, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(0.75, output(2).d_);

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_);
  EXPECT_FLOAT_EQ(    0, output(1).val_);
  EXPECT_FLOAT_EQ(  1.5, output(2).val_);
  EXPECT_FLOAT_EQ( -0.5, output(0).d_);
  EXPECT_FLOAT_EQ( -0.5, output(1).d_);
  EXPECT_FLOAT_EQ( -0.5, output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(2).val_);
  EXPECT_FLOAT_EQ(-25.5, output(0).d_);
  EXPECT_FLOAT_EQ( -0.5, output(1).d_);
  EXPECT_FLOAT_EQ( 0.25, output(2).d_);

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE (std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE (std::isnan(output(0).d_));
  EXPECT_TRUE (std::isnan(output(1).d_));
  EXPECT_TRUE (std::isnan(output(2).d_));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE (std::isnan(output(0).d_));
  EXPECT_TRUE (std::isnan(output(1).d_));
  EXPECT_TRUE (std::isnan(output(2).d_));
}
TEST(AgradFwdMatrix, divide_rowvector) {
  using stan::math::divide;
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
  
  row_vector_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  row_vector_fv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(2).val_);
  EXPECT_FLOAT_EQ( -25, output(0).d_);
  EXPECT_FLOAT_EQ(   0, output(1).d_);
  EXPECT_FLOAT_EQ(0.75, output(2).d_);

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_);
  EXPECT_FLOAT_EQ(    0, output(1).val_);
  EXPECT_FLOAT_EQ(  1.5, output(2).val_);
  EXPECT_FLOAT_EQ( -0.5, output(0).d_);
  EXPECT_FLOAT_EQ( -0.5, output(1).d_);
  EXPECT_FLOAT_EQ( -0.5, output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_);
  EXPECT_FLOAT_EQ(   0, output(1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(2).val_);
  EXPECT_FLOAT_EQ(-25.5, output(0).d_);
  EXPECT_FLOAT_EQ( -0.5, output(1).d_);
  EXPECT_FLOAT_EQ( 0.25, output(2).d_);

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE(std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE(std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE(std::isnan(output(0).d_));
  EXPECT_TRUE(std::isnan(output(1).d_));
  EXPECT_TRUE(std::isnan(output(2).d_));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE(std::isnan(output(0).d_));
  EXPECT_TRUE(std::isnan(output(1).d_));
  EXPECT_TRUE(std::isnan(output(2).d_));
}
TEST(AgradFwdMatrix, divide_matrix) {
  using stan::math::divide;
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
  
  matrix_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ(1.5, output_d(1,0));
  EXPECT_FLOAT_EQ( -2, output_d(1,1));

  matrix_fv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_);
  EXPECT_FLOAT_EQ( -25, output(0,0).d_);
  EXPECT_FLOAT_EQ(   0, output(0,1).d_);
  EXPECT_FLOAT_EQ(0.75, output(1,0).d_);
  EXPECT_FLOAT_EQ(  -1, output(1,1).d_);
  
  output = divide(v1, d2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_);
  EXPECT_FLOAT_EQ(   0, output(0,1).val_);
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_);
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_);
  EXPECT_FLOAT_EQ(-0.5, output(0,0).d_);
  EXPECT_FLOAT_EQ(-0.5, output(0,1).d_);
  EXPECT_FLOAT_EQ(-0.5, output(1,0).d_);
  EXPECT_FLOAT_EQ(-0.5, output(1,1).d_);
  
  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(  -50, output(0,0).val_);
  EXPECT_FLOAT_EQ(    0, output(0,1).val_);
  EXPECT_FLOAT_EQ(  1.5, output(1,0).val_);
  EXPECT_FLOAT_EQ(   -2, output(1,1).val_);
  EXPECT_FLOAT_EQ(-25.5, output(0,0).d_);
  EXPECT_FLOAT_EQ( -0.5, output(0,1).d_);
  EXPECT_FLOAT_EQ( 0.25, output(1,0).d_);
  EXPECT_FLOAT_EQ( -1.5, output(1,1).d_);

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0,0));
  EXPECT_TRUE(std::isnan(output_d(0,1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(1,0));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(1,1));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val_);
  EXPECT_TRUE (std::isnan(output(0,1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val_);
  EXPECT_TRUE (std::isnan(output(0,0).d_));
  EXPECT_TRUE (std::isnan(output(0,1).d_));
  EXPECT_TRUE (std::isnan(output(1,0).d_));
  EXPECT_TRUE (std::isnan(output(1,1).d_));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val_);
  EXPECT_TRUE (std::isnan(output(0,1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,1).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,0).d_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0,0).val_);
  EXPECT_TRUE (std::isnan(output(0,1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(1,0).val_);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(1,1).val_);
  EXPECT_TRUE (std::isnan(output(0,0).d_));
  EXPECT_TRUE (std::isnan(output(0,1).d_));
  EXPECT_TRUE (std::isnan(output(1,0).d_));
  EXPECT_TRUE (std::isnan(output(1,1).d_));
}
