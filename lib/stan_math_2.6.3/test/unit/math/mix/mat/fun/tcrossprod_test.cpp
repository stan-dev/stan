#include <stan/math/fwd/mat/fun/tcrossprod.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>

TEST(AgradMixMatrixTCrossProd, fv_3x3_matrix_1stDeriv) {
  using stan::math::matrix_fv;
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
  matrix_d X = Z * Z.transpose();
  matrix_fv output = stan::math::tcrossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(32,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(12,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(40,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(32,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(40,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val_,Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, fv_3x3_matrix_2ndDeriv) {
  using stan::math::matrix_fv;

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
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val_,Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, fv_2x2_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;
  matrix_fv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_d X = Z * Z.transpose();
  matrix_fv output = stan::math::tcrossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ(12,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ( 4,output(1,1).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, fv_2x2_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  matrix_fv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, fv_1x1_matrix_1stDeriv) {
  using stan::math::matrix_fv;

  matrix_fv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());

  AVEC q = createAVEC(Y(0,0).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
}
TEST(AgradMixMatrixTCrossProd, fv_1x1_matrix_2ndDeriv) {
  using stan::math::matrix_fv;

  matrix_fv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
}
TEST(AgradMixMatrixTCrossProd, fv_1x3_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
}
TEST(AgradMixMatrixTCrossProd, fv_1x3_matrix_2ndDeriv) {
  using stan::math::matrix_fv;

  matrix_fv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
}
TEST(AgradMixMatrixTCrossProd, fv_2x3_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 14, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-20, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-20, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ( 98, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 24, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(-24, output(1,1).d_.val()); 

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val_,Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, fv_2x3_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val_,Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, fv_3x2_matrix_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 5, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ( 1, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ( 1, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ(10, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(10, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(10, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ( 8, output(1,1).d_.val()); 

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val_,Y(1,1).val(),Y(2,0).val(),Y(2,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, fv_3x2_matrix_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val_,Y(1,1).val(),Y(2,0).val(),Y(2,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x3_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  matrix_ffv Y(3,3);
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
  matrix_d X = Z * Z.transpose();
  matrix_ffv output = stan::math::tcrossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(32,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(40,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(32,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(40,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x3_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(3,3);
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
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x3_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(3,3);
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
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x3_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(3,3);
  Y << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(2,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
   Y(2,2).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x2_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_d X = Z * Z.transpose();
  matrix_ffv output = stan::math::tcrossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ(12,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( 4,output(1,1).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x2_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x2_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x2_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x1_matrix_1stDeriv) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x1_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x1_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x1_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x3_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x3_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x3_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
}
TEST(AgradMixMatrixTCrossProd, ffv_1x3_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x3_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 14, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(-20, output(0,1).val_.val().val()); 
  EXPECT_FLOAT_EQ(-20, output(1,0).val_.val().val()); 
  EXPECT_FLOAT_EQ( 98, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( 24, output(0,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val().val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(-24, output(1,1).d_.val().val()); 

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x3_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x3_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_2x3_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(1,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val_.val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x2_matrix_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 5, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ( 1, output(0,1).val_.val().val()); 
  EXPECT_FLOAT_EQ( 1, output(1,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(10, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(10, output(0,1).d_.val().val()); 
  EXPECT_FLOAT_EQ(10, output(1,0).d_.val().val()); 
  EXPECT_FLOAT_EQ( 8, output(1,1).d_.val().val()); 

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val_.val(),Y(1,1).val().val(),Y(2,0).val().val(),Y(2,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradMixMatrixTCrossProd, ffv_3x2_matrix_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val_.val(),Y(1,1).val().val(),Y(2,0).val().val(),Y(2,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}

TEST(AgradMixMatrixTCrossProd, ffv_3x2_matrix_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val_.val(),Y(1,1).val().val(),Y(2,0).val().val(),Y(2,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}

TEST(AgradMixMatrixTCrossProd, ffv_3x2_matrix_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(1,0).d_ = 1.0;
   Y(1,1).d_ = 1.0;
   Y(2,0).d_ = 1.0;
   Y(2,1).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
  matrix_ffv output = stan::math::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val_.val(),Y(1,1).val().val(),Y(2,0).val().val(),Y(2,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
