#include <stan/math/fwd/mat/fun/divide.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/divide.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>

using stan::math::fvar;
using stan::math::var;
TEST(AgradMixMatrixOperatorDivision,fv_scalar_1stDeriv) {
  using stan::math::divide;
  double d1, d2;
  fvar<var> v1, v2;

  d1 = 10;
  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  v2 = -2;
  v2.d_ = 1.0;
  
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

  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val());
  VEC h;
  divide(v1,d2).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,fv_scalar_2ndDeriv) {
  using stan::math::divide;
  double d2;
  fvar<var> v1, v2;

  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val());
  VEC h;
  divide(v1,d2).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,fv_vector_1stDeriv) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_d d1(3);
  vector_fv v1(3);
  double d2;
  fvar<var> v2;
  
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
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val());
  EXPECT_TRUE (std::isnan(output(1).val_.val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val());
  EXPECT_TRUE (std::isnan(output(0).d_.val()));
  EXPECT_TRUE (std::isnan(output(1).d_.val()));
  EXPECT_TRUE (std::isnan(output(2).d_.val()));

  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,fv_vector_2ndDeriv) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  vector_fv v1(3);
  fvar<var> v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  vector_fv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,fv_rowvector_1stDeriv) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(3);
  row_vector_fv v1(3);
  double d2;
  fvar<var> v2;
  
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

  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,fv_rowvector_2ndDeriv) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_fv v1(3);
  fvar<var> v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  row_vector_fv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,fv_matrix_1stDeriv) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

  matrix_d d1(2,2);
  matrix_fv v1(2,2);
  double d2;
  fvar<var> v2;
  
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

  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorDivision,fv_matrix_2ndDeriv) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

  matrix_fv v1(2,2);
  fvar<var> v2;
  
  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  matrix_fv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_scalar_1stDeriv) {
  using stan::math::divide;
  double d1, d2;
  fvar<fvar<var> > v1, v2;

  d1 = 10;
  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  v2 = -2;
  v2.d_ = 1.0;
  
  EXPECT_FLOAT_EQ(  -5, divide(d1, d2));
  EXPECT_FLOAT_EQ(  -5, divide(d1, v2).val_.val().val());
  EXPECT_FLOAT_EQ(  -5, divide(v1, d2).val_.val().val());
  EXPECT_FLOAT_EQ(  -5, divide(v1, v2).val_.val().val());
  EXPECT_FLOAT_EQ(-2.5, divide(d1, v2).d_.val().val());
  EXPECT_FLOAT_EQ(-0.5, divide(v1, d2).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, divide(v1, v2).d_.val().val());

  d2 = 0;
  v2 = 0;

  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), divide(d1, d2));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(d1, v2).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, d2).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, v2).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  divide(v1, d2).d_.val().val());
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_.val().val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_.val().val()));

  d1 = 0;
  v1 = 0;
  EXPECT_TRUE(std::isnan(divide(d1, d2)));
  EXPECT_TRUE(std::isnan(divide(d1, v2).val_.val().val()));
  EXPECT_TRUE(std::isnan(divide(v1, d2).val_.val().val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).val_.val().val()));
  EXPECT_TRUE(std::isnan(divide(d1, v2).d_.val().val()));
  EXPECT_TRUE(std::isnan(divide(v1, d2).d_.val().val()));
  EXPECT_TRUE(std::isnan(divide(v1, v2).d_.val().val()));

  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val().val());
  VEC h;
  divide(v1,d2).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_scalar_2ndDeriv_1) {
  using stan::math::divide;
  double d2;
  fvar<fvar<var> > v1, v2;

  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val().val());
  VEC h;
  divide(v1,d2).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_scalar_2ndDeriv_2) {
  using stan::math::divide;
  double d2;
  fvar<fvar<var> > v1, v2;

  v1 = 10;
  v1.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val().val());
  VEC h;
  divide(v1,d2).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_scalar_3rdDeriv) {
  using stan::math::divide;
  double d2;
  fvar<fvar<var> > v1, v2;

  v1 = 10;
  v1.d_ = 1.0;
  v1.val_.d_ = 1.0;
  d2 = -2;
  
  AVEC q = createAVEC(v1.val().val());
  VEC h;
  divide(v1,d2).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_vector_1stDeriv) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  double d2;
  fvar<fvar<var> > v2;
  
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

  vector_ffv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( -25, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(0.75, output(2).d_.val().val());

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(    0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(  1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(2).d_.val().val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(-25.5, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 0.25, output(2).d_.val().val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE (std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(2).d_.val().val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_.val().val());
  EXPECT_TRUE (std::isnan(output(1).d_.val().val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_.val().val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(2).d_.val().val()));

  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_vector_2ndDeriv_1) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_vector_2ndDeriv_2) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_vector_3rdDeriv) {
  using stan::math::divide;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 1.0;
  v1(2).val_.d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  v2.val_.d_ = 1.0;
  vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_rowvector_1stDeriv) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  double d2;
  fvar<fvar<var> > v2;
  
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

  row_vector_ffv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( -25, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(0.75, output(2).d_.val().val());

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(  -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(    0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(  1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(2).d_.val().val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ( -50, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(-25.5, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 0.25, output(2).d_.val().val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0));
  EXPECT_TRUE(std::isnan(output_d(1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(2));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE(std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_TRUE(std::isnan(output(0).d_.val().val()));
  EXPECT_TRUE(std::isnan(output(1).d_.val().val()));
  EXPECT_TRUE(std::isnan(output(2).d_.val().val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(0).d_.val().val());
  EXPECT_TRUE (std::isnan(output(1).d_.val().val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output(2).d_.val().val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(2).val_.val().val());
  EXPECT_TRUE(std::isnan(output(0).d_.val().val()));
  EXPECT_TRUE(std::isnan(output(1).d_.val().val()));
  EXPECT_TRUE(std::isnan(output(2).d_.val().val()));

  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_rowvector_2ndDeriv_1) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  row_vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_rowvector_2ndDeriv_2) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  row_vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_rowvector_3rdDeriv) {
  using stan::math::divide;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_ffv v1(3);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 1.0;
  v1(2).val_.d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  v2.val_.d_ = 1.0;

  row_vector_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_matrix_1stDeriv) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,2);
  matrix_ffv v1(2,2);
  double d2;
  fvar<fvar<var> > v2;
  
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

  matrix_ffv output;
  output = divide(d1, v2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( -25, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(0.75, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -1, output(1,1).d_.val().val());
  
  output = divide(v1, d2);
  EXPECT_FLOAT_EQ( -50, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 1.5, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(-0.5, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(-0.5, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(-0.5, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(-0.5, output(1,1).d_.val().val());
  
  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(  -50, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(    0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(  1.5, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(   -2, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(-25.5, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( -0.5, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( 0.25, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( -1.5, output(1,1).d_.val().val());

  d2 = 0;
  v2 = 0;
  output_d = divide(d1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(0,0));
  EXPECT_TRUE(std::isnan(output_d(0,1)));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), output_d(1,0));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), output_d(1,1));

  output = divide(d1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0,0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(0,1).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1,0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1,1).d_.val().val()));

  output = divide(v1, d2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),
                  output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).d_.val().val());
  EXPECT_TRUE (std::isnan(output(0,1).d_.val().val()));
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),
                  output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(1,1).d_.val().val());

  output = divide(v1, v2);
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(), 
                  output(0,0).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0,1).val_.val().val()));
  EXPECT_FLOAT_EQ(-std::numeric_limits<double>::infinity(), 
                  output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(std::numeric_limits<double>::infinity(),
                  output(1,1).val_.val().val());
  EXPECT_TRUE (std::isnan(output(0,0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(0,1).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1,0).d_.val().val()));
  EXPECT_TRUE (std::isnan(output(1,1).d_.val().val()));

  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.5,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorDivision,ffv_matrix_2ndDeriv_1) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_ffv v1(2,2);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  matrix_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradMixMatrixOperatorDivision,ffv_matrix_2ndDeriv_2) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_ffv v1(2,2);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;

  matrix_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradMixMatrixOperatorDivision,ffv_matrix_3rdDeriv) {
  using stan::math::divide;
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_ffv v1(2,2);
  fvar<fvar<var> > v2;
  
  v1 << 100, 0, -3, 4;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v1(0,0).val_.d_ = 1.0;
  v1(0,1).val_.d_ = 1.0;
  v1(1,0).val_.d_ = 1.0;
  v1(1,1).val_.d_ = 1.0;
  v2 = -2;
  v2.d_ = 1.0;
  v2.val_.d_ = 1.0;

  matrix_ffv output;

  output = divide(v1, v2);
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.25,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
