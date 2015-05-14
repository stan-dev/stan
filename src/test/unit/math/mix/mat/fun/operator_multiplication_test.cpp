#include <gtest/gtest.h>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/core.hpp>

using stan::math::fvar;  
using stan::math::var;  
using stan::math::multiply;

TEST(AgradMixMatrixOperatorMultiplication,fv_vector_scalar_1stDeriv) {
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

  vector_fv output;
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

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_vector_scalar_2ndDeriv) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_scalar_1stDeriv) {
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
  
  row_vector_fv output;
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

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_scalar_2ndDeriv) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  output(0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_scalar_1stDeriv) {
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

  matrix_fv output;
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

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_scalar_2ndDeriv) {
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
  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  EXPECT_FLOAT_EQ( 3, multiply(v1, v2).val_.val());
  EXPECT_FLOAT_EQ( 3, multiply(v1, d2).val_.val());
  EXPECT_FLOAT_EQ( 3, multiply(d1, v2).val_.val());
  EXPECT_FLOAT_EQ( 0, multiply(v1, v2).d_.val());
  EXPECT_FLOAT_EQ( 1, multiply(v1 ,d2).d_.val());
  EXPECT_FLOAT_EQ(-1, multiply(d1 ,v2).d_.val());

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  multiply(v1 ,d2).val_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(-1,h[2]);

  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_vector_2ndDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  AVEC q = createAVEC(v1(0).val(),v1(1).val(),v1(2).val());
  VEC h;
  multiply(v1 ,d2).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_vector_rowvector_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  AVEC q = createAVEC(v2(0).val(),v2(1).val(),v2(2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_vector_rowvector_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  matrix_fv output;
  output = multiply(d1, v2);


  AVEC q = createAVEC(v2(0).val(),v2(1).val(),v2(2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_vector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;

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

  output = multiply(v1, d2);
  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_vector_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;

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

  vector_fv output;
  output = multiply(v1, d2);

  AVEC q = createAVEC(v1(0,0).val(),v1(0,1).val(),v1(1,0).val(),v1(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_d;
  using stan::math::vector_fv;

  matrix_d d1(3,2);
  matrix_fv v1(3,2);
  vector_d d2(4);
  vector_fv v2(4);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  AVEC q = createAVEC(v2(0,0).val(),v2(0,1).val(),v2(1,0).val(),v2(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::vector_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

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

  vector_fv output;
  output = multiply(d1, v2);

  AVEC q = createAVEC(v2(0,0).val(),v2(0,1).val(),v2(1,0).val(),v2(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_fv;

  row_vector_d d1(4);
  row_vector_fv v1(4);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

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

  AVEC q = createAVEC(v2(0,0).val(),v2(0,1).val(),v2(1,0).val(),v2(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(9,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(24,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_matrix_2ndDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

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

  matrix_fv output;
  output = multiply(d1, v2);
  AVEC q = createAVEC(v2(0,0).val(),v2(0,1).val(),v2(1,0).val(),v2(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,fv_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_fv;

  matrix_d d1(2,2);
  matrix_fv v1(2,2);
  matrix_d d2(3,2);
  matrix_fv v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_scalar_1stDeriv) {
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

  vector_ffv output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val().val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val().val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val().val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val().val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val().val());

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_scalar_2ndDeriv_1) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_scalar_2ndDeriv_2) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_scalar_3rdDeriv) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}

TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_scalar_1stDeriv) {
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
  
  row_vector_ffv output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val().val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val().val());

  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val().val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(2).d_.val().val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(2).d_.val().val());

  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(2).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(2).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_scalar_2ndDeriv_1) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_scalar_2ndDeriv_2) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_scalar_3rdDeriv) {
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

  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  output(0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_scalar_1stDeriv) {
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

  matrix_ffv output;
  output = multiply(d1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(   4, output(1,1).d_.val().val());

  output = multiply(v1, d2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_.val().val());
 
  output = multiply(v1, v2);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(   2, output(1,1).d_.val().val());

  output = multiply(v2, d1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( 100, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -3, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(   4, output(1,1).d_.val().val());

  output = multiply(d2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,1).d_.val().val());
 
  output = multiply(v2, v1);
  EXPECT_FLOAT_EQ(-200, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(   0, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ(  -8, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  98, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -5, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(   2, output(1,1).d_.val().val());

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_scalar_2ndDeriv_1) {
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
  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_scalar_2ndDeriv_2) {
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
  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_scalar_3rdDeriv) {
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
  output = multiply(v2, v1);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_vector_1stDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);
  
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

  EXPECT_FLOAT_EQ( 3, multiply(v1, v2).val_.val().val());
  EXPECT_FLOAT_EQ( 3, multiply(v1, d2).val_.val().val());
  EXPECT_FLOAT_EQ( 3, multiply(d1, v2).val_.val().val());
  EXPECT_FLOAT_EQ( 0, multiply(v1, v2).d_.val().val());
  EXPECT_FLOAT_EQ( 1, multiply(v1 ,d2).d_.val().val());
  EXPECT_FLOAT_EQ(-1, multiply(d1 ,v2).d_.val().val());

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  multiply(v1 ,d2).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(-2,h[1]);
  EXPECT_FLOAT_EQ(-1,h[2]);

  d1.resize(1);
  v1.resize(1);
  EXPECT_THROW(multiply(v1, v2), std::domain_error);
  EXPECT_THROW(multiply(v1, d2), std::domain_error);
  EXPECT_THROW(multiply(d1, v2), std::domain_error);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_vector_2ndDeriv_1) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);
  
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

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  multiply(v1 ,d2).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_vector_2ndDeriv_2) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);
  
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

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  multiply(v1 ,d2).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_vector_3rdDeriv) {
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  vector_d d2(3);
  vector_ffv v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 1.0;
  v1(2).val_.d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  v2(2).d_ = 1.0;
  v2(0).val_.d_ = 1.0;
  v2(1).val_.d_ = 1.0;
  v2(2).val_.d_ = 1.0;

  AVEC q = createAVEC(v1(0).val().val(),v1(1).val().val(),v1(2).val().val());
  VEC h;
  multiply(v1 ,d2).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_rowvector_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);
  
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

  matrix_ffv output = multiply(v1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val().val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(  5, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  0, output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(  7, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  1, output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(  2, output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ( -7, output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ( -6, output(2,2).d_.val().val());
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val().val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(  4, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( -2, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(  4, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( -2, output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(  4, output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ( -2, output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ( -1, output(2,2).d_.val().val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.rows());
  EXPECT_EQ(3, output.cols());
  EXPECT_FLOAT_EQ(  4, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( -2, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( -1, output(0,2).val_.val().val());
  EXPECT_FLOAT_EQ( 12, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( -6, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( -3, output(1,2).val_.val().val());
  EXPECT_FLOAT_EQ(-20, output(2,0).val_.val().val());  
  EXPECT_FLOAT_EQ( 10, output(2,1).val_.val().val());
  EXPECT_FLOAT_EQ(  5, output(2,2).val_.val().val());
  EXPECT_FLOAT_EQ(  1, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  1, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  1, output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(  3, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  3, output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(  3, output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ( -5, output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ( -5, output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ( -5, output(2,2).d_.val().val());

  AVEC q = createAVEC(v2(0).val().val(),v2(1).val().val(),v2(2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(1,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_rowvector_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);
  
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

  matrix_ffv output;
  output = multiply(d1, v2);


  AVEC q = createAVEC(v2(0).val().val(),v2(1).val().val(),v2(2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_rowvector_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);
  
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

  matrix_ffv output;
  output = multiply(d1, v2);


  AVEC q = createAVEC(v2(0).val().val(),v2(1).val().val(),v2(2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_vector_rowvector_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  vector_d d1(3);
  vector_ffv v1(3);
  row_vector_d d2(3);
  row_vector_ffv v2(3);
  
  d1 << 1, 3, -5;
  v1 << 1, 3, -5;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 1.0;
  v1(2).val_.d_ = 1.0;
  d2 << 4, -2, -1;
  v2 << 4, -2, -1;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  v2(2).d_ = 1.0;
  v2(0).val_.d_ = 1.0;
  v2(1).val_.d_ = 1.0;
  v2(2).val_.d_ = 1.0;

  matrix_ffv output;
  output = multiply(d1, v2);


  AVEC q = createAVEC(v2(0).val().val(),v2(1).val().val(),v2(2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_vector_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  matrix_d d1(3,2);
  matrix_ffv v1(3,2);
  vector_d d2(2);
  vector_ffv v2(2);
  
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

  vector_ffv output = multiply(v1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 6, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(2).d_.val().val());
  
  output = multiply(v1, d2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 2, output(0).d_.val().val());
  EXPECT_FLOAT_EQ( 2, output(1).d_.val().val());
  EXPECT_FLOAT_EQ( 2, output(2).d_.val().val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(3, output.size());
  EXPECT_FLOAT_EQ(10, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(26, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(2).val_.val().val());
  EXPECT_FLOAT_EQ( 4, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(-1, output(1).d_.val().val());
  EXPECT_FLOAT_EQ(-3, output(2).d_.val().val());

  output = multiply(v1, d2);
  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_vector_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  matrix_d d1(3,2);
  matrix_ffv v1(3,2);
  vector_d d2(2);
  vector_ffv v2(2);
  
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

  vector_ffv output;
  output = multiply(v1, d2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_vector_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  matrix_d d1(3,2);
  matrix_ffv v1(3,2);
  vector_d d2(2);
  vector_ffv v2(2);
  
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

  vector_ffv output;
  output = multiply(v1, d2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_vector_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  matrix_d d1(3,2);
  matrix_ffv v1(3,2);
  vector_d d2(2);
  vector_ffv v2(2);
  
  d1 << 1, 3, -5, 4, -2, -1;
  v1 << 1, 3, -5, 4, -2, -1;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v1(2,0).d_ = 1.0;
  v1(2,1).d_ = 1.0;
  v1(0,0).val_.d_ = 1.0;
  v1(0,1).val_.d_ = 1.0;
  v1(1,0).val_.d_ = 1.0;
  v1(1,1).val_.d_ = 1.0;
  v1(2,0).val_.d_ = 1.0;
  v1(2,1).val_.d_ = 1.0;
  d2 << -2, 4;
  v2 << -2, 4;
  v2(0).d_ = 1.0;
  v2(1).d_ = 1.0;
  v2(0).val_.d_ = 1.0;
  v2(1).val_.d_ = 1.0;

  vector_ffv output;
  output = multiply(v1, d2);

  AVEC q = createAVEC(v1(0,0).val().val(),v1(0,1).val().val(),v1(1,0).val().val(),v1(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_vector_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_d;
  using stan::math::vector_ffv;

  matrix_d d1(3,2);
  matrix_ffv v1(3,2);
  vector_d d2(4);
  vector_ffv v2(4);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  vector_ffv output = multiply(v1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( -3, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  9, output(1).d_.val().val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val().val());
  EXPECT_FLOAT_EQ( -6, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  6, output(1).d_.val().val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.size());
  EXPECT_FLOAT_EQ(-24, output(0).val_.val().val());
  EXPECT_FLOAT_EQ(  9, output(1).val_.val().val());
  EXPECT_FLOAT_EQ(  3, output(0).d_.val().val());
  EXPECT_FLOAT_EQ(  3, output(1).d_.val().val());

  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  vector_ffv output;
  output = multiply(d1, v2);

  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  vector_ffv output;
  output = multiply(d1, v2);

  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(3);
  row_vector_ffv v1(3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
  d1 << -2, 4, 1;
  v1 << -2, 4, 1;
  v1(0).d_ = 1.0;
  v1(1).d_ = 1.0;
  v1(2).d_ = 1.0;
  v1(0).val_.d_ = 1.0;
  v1(1).val_.d_ = 1.0;
  v1(2).val_.d_ = 1.0;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;
  v2(0,0).d_ = 1.0;
  v2(0,1).d_ = 1.0;
  v2(1,0).d_ = 1.0;
  v2(1,1).d_ = 1.0;
  v2(2,0).d_ = 1.0;
  v2(2,1).d_ = 1.0;
  v2(0,0).val_.d_ = 1.0;
  v2(0,1).val_.d_ = 1.0;
  v2(1,0).val_.d_ = 1.0;
  v2(1,1).val_.d_ = 1.0;
  v2(2,0).val_.d_ = 1.0;
  v2(2,1).val_.d_ = 1.0;


  vector_ffv output;
  output = multiply(d1, v2);

  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_rowvector_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;
  using stan::math::row_vector_d;
  using stan::math::row_vector_ffv;

  row_vector_d d1(4);
  row_vector_ffv v1(4);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_matrix_1stDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,3);
  matrix_ffv v1(2,3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  matrix_ffv output = multiply(v1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  30, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  42, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -2, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(  10, output(1,1).d_.val().val());

  output = multiply(v1, d2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(  -6, output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(   6, output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  -6, output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(   6, output(1,1).d_.val().val());
  
  output = multiply(d1, v2);
  EXPECT_EQ(2, output.rows());
  EXPECT_EQ(2, output.cols());
  EXPECT_FLOAT_EQ(-117, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ( 120, output(0,1).val_.val().val());
  EXPECT_FLOAT_EQ( 157, output(1,0).val_.val().val());
  EXPECT_FLOAT_EQ( 135, output(1,1).val_.val().val());

  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(9,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(24,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_matrix_2ndDeriv_1) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,3);
  matrix_ffv v1(2,3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  matrix_ffv output;
  output = multiply(d1, v2);
  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_matrix_2ndDeriv_2) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,3);
  matrix_ffv v1(2,3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
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

  matrix_ffv output;
  output = multiply(d1, v2);
  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_matrix_3rdDeriv) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,3);
  matrix_ffv v1(2,3);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);
  
  d1 << 9, 24, 3, 46, -9, -33;
  v1 << 9, 24, 3, 46, -9, -33;
  v1(0,0).d_ = 1.0;
  v1(0,1).d_ = 1.0;
  v1(0,2).d_ = 1.0;
  v1(1,0).d_ = 1.0;
  v1(1,1).d_ = 1.0;
  v1(1,2).d_ = 1.0;
  v1(0,0).val_.d_ = 1.0;
  v1(0,1).val_.d_ = 1.0;
  v1(0,2).val_.d_ = 1.0;
  v1(1,0).val_.d_ = 1.0;
  v1(1,1).val_.d_ = 1.0;
  v1(1,2).val_.d_ = 1.0;
  d2 << 1, 3, -5, 4, -2, -1;
  v2 << 1, 3, -5, 4, -2, -1;
  v2(0,0).d_ = 1.0;
  v2(0,1).d_ = 1.0;
  v2(1,0).d_ = 1.0;
  v2(1,1).d_ = 1.0;
  v2(2,0).d_ = 1.0;
  v2(2,1).d_ = 1.0;
  v2(0,0).val_.d_ = 1.0;
  v2(0,1).val_.d_ = 1.0;
  v2(1,0).val_.d_ = 1.0;
  v2(1,1).val_.d_ = 1.0;
  v2(2,0).val_.d_ = 1.0;
  v2(2,1).val_.d_ = 1.0;


  matrix_ffv output;
  output = multiply(d1, v2);
  AVEC q = createAVEC(v2(0,0).val().val(),v2(0,1).val().val(),v2(1,0).val().val(),v2(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixOperatorMultiplication,ffv_matrix_matrix_exception) {
  using stan::math::matrix_d;
  using stan::math::matrix_ffv;

  matrix_d d1(2,2);
  matrix_ffv v1(2,2);
  matrix_d d2(3,2);
  matrix_ffv v2(3,2);

  EXPECT_THROW(multiply(v1, v2), std::invalid_argument);
  EXPECT_THROW(multiply(v1, d2), std::invalid_argument);
  EXPECT_THROW(multiply(d1, v2), std::invalid_argument);
}
