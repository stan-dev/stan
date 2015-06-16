#include <stan/math/fwd/mat/fun/divide.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/divide.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>

using stan::math::fvar;
TEST(AgradFwdMatrixOperatorDivision,fd_scalar) {
  using stan::math::divide;
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
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(v1, d2).d_);
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_));
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
TEST(AgradFwdMatrixOperatorDivision,fd_vector) {
  using stan::math::divide;
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
  
  vector_d output_d;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  vector_fd output;
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
  EXPECT_TRUE (std::isnan(output(1).d_));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE (std::isnan(output(0).d_));
  EXPECT_TRUE (std::isnan(output(1).d_));
  EXPECT_TRUE (std::isnan(output(2).d_));
}
TEST(AgradFwdMatrixOperatorDivision,fd_rowvector) {
  using stan::math::divide;
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
  
  row_vector_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  row_vector_fd output;
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
  EXPECT_TRUE (std::isnan(output(1).d_));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_);

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_);
  EXPECT_TRUE (std::isnan(output(1).val_));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_);
  EXPECT_TRUE(std::isnan(output(0).d_));
  EXPECT_TRUE(std::isnan(output(1).d_));
  EXPECT_TRUE(std::isnan(output(2).d_));
}
TEST(AgradFwdMatrixOperatorDivision,fd_matrix) {
  using stan::math::divide;
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
  
  matrix_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ(1.5, output_d(1,0));
  EXPECT_FLOAT_EQ( -2, output_d(1,1));

  matrix_fd output;
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
  EXPECT_TRUE (std::isnan(output(0,1).d_));
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
TEST(AgradFwdMatrixOperatorDivision,ffd_scalar) {
  using stan::math::divide;
  double d1, d2;
  fvar<fvar<double> > v1, v2;

  d1 = 10;
  v1.val_.val_ = 10;
  v1.d_.val_ = 1.0;
  d2 = -2;
  v2.val_.val_ = -2;
  v2.d_.val_ = 1.0;
  
  EXPECT_FLOAT_EQ(  -5, divide(d1, d2));
  EXPECT_FLOAT_EQ(  -5, divide(d1, v2).val_.val());
  EXPECT_FLOAT_EQ(  -5, divide(v1, d2).val_.val());
  EXPECT_FLOAT_EQ(  -5, divide(v1, v2).val_.val());
  EXPECT_FLOAT_EQ(-2.5, divide(d1, v2).d_.val());
  EXPECT_FLOAT_EQ(-0.5, divide(v1, d2).d_.val());
  EXPECT_FLOAT_EQ(  -3, divide(v1, v2).d_.val());

  d2 = 0;
  v2 = 0;

  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(d1, d2));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(d1, v2).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, d2).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, v2).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, d2).d_.val());
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_.val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_.val()));

  d1 = 0;
  v1 = 0;
  EXPECT_TRUE(std::isnan(divide(d1, d2)));
  EXPECT_TRUE(std::isnan(divide(d1, v2).val_.val()));
  EXPECT_TRUE(std::isnan(divide(v1, d2).val_.val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).val_.val()));
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_.val()));
  EXPECT_TRUE(std::isnan(divide(v1, d2).d_.val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_.val()));
}
TEST(AgradFwdMatrixOperatorDivision,ffd_vector) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_ffd;

  vector_d d1(3);
  vector_ffd v1(3);
  double d2;
  fvar<fvar<double> > v2,a,b,c;
  
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
  
  vector_d output_d;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  vector_ffd output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ( -25, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(0.75, output(2).d_.val());

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(    0, output(1).val_.val());
  EXPECT_FLOAT_EQ(  1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ( -0.5, output(0).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(2).d_.val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ(-25.5, output(0).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ( 0.25, output(2).d_.val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE (std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val());
  EXPECT_TRUE (std::isnan(output(0).d_.val()));
  EXPECT_TRUE (std::isnan(output(1).d_.val()));
  EXPECT_TRUE (std::isnan(output(2).d_.val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_.val());
  EXPECT_TRUE (std::isnan(output(1).d_.val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_.val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output(2).val_.val());
  EXPECT_TRUE (std::isnan(output(0).d_.val()));
  EXPECT_TRUE (std::isnan(output(1).d_.val()));
  EXPECT_TRUE (std::isnan(output(2).d_.val()));
}
TEST(AgradFwdMatrixOperatorDivision,ffd_rowvector) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffd;

  row_vector_d d1(3);
  row_vector_ffd v1(3);
  double d2;
  fvar<fvar<double> > v2,a,b,c;
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
  
  row_vector_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0));
  EXPECT_FLOAT_EQ(  0, output_d(1));
  EXPECT_FLOAT_EQ(1.5, output_d(2));

  row_vector_ffd output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ( -25, output(0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val());
  EXPECT_FLOAT_EQ(0.75, output(2).d_.val());

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(    0, output(1).val_.val());
  EXPECT_FLOAT_EQ(  1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ( -0.5, output(0).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(2).d_.val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val());
  EXPECT_FLOAT_EQ(-25.5, output(0).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val());
  EXPECT_FLOAT_EQ( 0.25, output(2).d_.val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE(std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE(std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val());
  EXPECT_TRUE(std::isnan(output(0).d_.val()));
  EXPECT_TRUE(std::isnan(output(1).d_.val()));
  EXPECT_TRUE(std::isnan(output(2).d_.val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(),
                  output(2).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_.val());
  EXPECT_TRUE (std::isnan(output(1).d_.val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_.val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val());
  EXPECT_TRUE(std::isnan(output(0).d_.val()));
  EXPECT_TRUE(std::isnan(output(1).d_.val()));
  EXPECT_TRUE(std::isnan(output(2).d_.val()));
}
TEST(AgradFwdMatrixOperatorDivision,ffd_matrix) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_ffd;

  matrix_d d1(2,2);
  matrix_ffd v1(2,2);
  double d2;
  fvar<fvar<double> > v2,a,b,c,d;
  
  a.val_.val_ = 100.0;
  b.val_.val_ = 0.0;
  c.val_.val_ = -3.0;
  d.val_.val_ = 4.0;
  a.d_.val_ = 1.0;
  b.d_.val_ = 1.0;
  c.d_.val_ = 1.0;
  d.d_.val_ = 1.0;
  d1 << 100, 0, -3, 4;
  v1 << a,b,c,d;
  d2 = -2;
  v2.val_.val_ = -2;
  v2.d_.val_ = 1.0;
  
  matrix_d output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(-50, output_d(0,0));
  EXPECT_FLOAT_EQ(  0, output_d(0,1));
  EXPECT_FLOAT_EQ(1.5, output_d(1,0));
  EXPECT_FLOAT_EQ( -2, output_d(1,1));

  matrix_ffd output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( -25, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(0.75, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(  -1, output(1,1).d_.val());
  
  output = divide(v1, d2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(-0.5, output(0,0).d_.val());
  EXPECT_FLOAT_EQ(-0.5, output(0,1).d_.val());
  EXPECT_FLOAT_EQ(-0.5, output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-0.5, output(1,1).d_.val());
  
  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(  -50, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(    0, output(0,1).val_.val());
  EXPECT_FLOAT_EQ(  1.5, output(1,0).val_.val());
  EXPECT_FLOAT_EQ(   -2, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(-25.5, output(0,0).d_.val());
  EXPECT_FLOAT_EQ( -0.5, output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 0.25, output(1,0).d_.val());
  EXPECT_FLOAT_EQ( -1.5, output(1,1).d_.val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0,0));
  EXPECT_TRUE(std::isnan(output_d(0,1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(1,0));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(1,1));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).val_.val());
  EXPECT_TRUE (std::isnan(output(0,0).d_.val()));
  EXPECT_TRUE (std::isnan(output(0,1).d_.val()));
  EXPECT_TRUE (std::isnan(output(1,0).d_.val()));
  EXPECT_TRUE (std::isnan(output(1,1).d_.val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).d_.val());
  EXPECT_TRUE (std::isnan(output(0,1).d_.val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,0).d_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).d_.val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).val_.val());
  EXPECT_TRUE (std::isnan(output(0,0).d_.val()));
  EXPECT_TRUE (std::isnan(output(0,1).d_.val()));
  EXPECT_TRUE (std::isnan(output(1,0).d_.val()));
  EXPECT_TRUE (std::isnan(output(1,1).d_.val()));
}
