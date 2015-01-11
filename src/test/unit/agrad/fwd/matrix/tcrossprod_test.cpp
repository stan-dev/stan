#include <stan/agrad/fwd/matrix/tcrossprod.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>

TEST(AgradFwdMatrixTCrossProd, fd_3x3_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  matrix_fd Y(3,3);
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
  matrix_fd output = stan::agrad::tcrossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_);
  EXPECT_FLOAT_EQ(12,output(0,1).d_);
  EXPECT_FLOAT_EQ(32,output(0,2).d_);
  EXPECT_FLOAT_EQ(12,output(1,0).d_);
  EXPECT_FLOAT_EQ(20,output(1,1).d_);
  EXPECT_FLOAT_EQ(40,output(1,2).d_);
  EXPECT_FLOAT_EQ(32,output(2,0).d_);
  EXPECT_FLOAT_EQ(40,output(2,1).d_);
  EXPECT_FLOAT_EQ(60,output(2,2).d_);
}
TEST(AgradFwdMatrixTCrossProd, fd_2x2_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;
  matrix_fd Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_d X = Z * Z.transpose();
  matrix_fd output = stan::agrad::tcrossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_);
  }

  EXPECT_FLOAT_EQ(12,output(0,0).d_);
  EXPECT_FLOAT_EQ( 8,output(0,1).d_);
  EXPECT_FLOAT_EQ( 8,output(1,0).d_);
  EXPECT_FLOAT_EQ( 4,output(1,1).d_);
}
TEST(AgradFwdMatrixTCrossProd, fd_1x1_matrix) {
  using stan::agrad::matrix_fd;

  matrix_fd Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fd output = stan::agrad::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_);
  EXPECT_FLOAT_EQ(12, output(0,0).d_);
}
TEST(AgradFwdMatrixTCrossProd, fd_1x3_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;

  matrix_fd Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_);
}
TEST(AgradFwdMatrixTCrossProd, fd_2x3_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;

  matrix_fd Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_fd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 14, output(0,0).val_); 
  EXPECT_FLOAT_EQ(-20, output(0,1).val_); 
  EXPECT_FLOAT_EQ(-20, output(1,0).val_); 
  EXPECT_FLOAT_EQ( 98, output(1,1).val_);
  EXPECT_FLOAT_EQ( 24, output(0,0).d_); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_); 
  EXPECT_FLOAT_EQ(-24, output(1,1).d_); 
}
TEST(AgradFwdMatrixTCrossProd, fd_3x2_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;

  matrix_fd Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 5, output(0,0).val_); 
  EXPECT_FLOAT_EQ( 1, output(0,1).val_); 
  EXPECT_FLOAT_EQ( 1, output(1,0).val_); 
  EXPECT_FLOAT_EQ(10, output(1,1).val_);
  EXPECT_FLOAT_EQ(12, output(0,0).d_); 
  EXPECT_FLOAT_EQ(10, output(0,1).d_); 
  EXPECT_FLOAT_EQ(10, output(1,0).d_); 
  EXPECT_FLOAT_EQ( 8, output(1,1).d_); 
}
TEST(AgradFwdMatrixTCrossProd, fv_3x3_matrix_1stDeriv) {
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
  matrix_d X = Z * Z.transpose();
  matrix_fv output = stan::agrad::tcrossprod(Y);
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
TEST(AgradFwdMatrixTCrossProd, fv_3x3_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;

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
  matrix_fv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, fv_2x2_matrix_1stDeriv) {
  using stan::agrad::matrix_fv;
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
  matrix_fv output = stan::agrad::tcrossprod(Y);
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
TEST(AgradFwdMatrixTCrossProd, fv_2x2_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;
  matrix_fv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_fv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixTCrossProd, fv_1x1_matrix_1stDeriv) {
  using stan::agrad::matrix_fv;

  matrix_fv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fv output = stan::agrad::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());

  AVEC q = createAVEC(Y(0,0).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, fv_1x1_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;

  matrix_fv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_fv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, fv_1x3_matrix_1stDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;

  matrix_fv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fv output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, fv_1x3_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;

  matrix_fv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_fv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, fv_2x3_matrix_1stDeriv) {
  using stan::agrad::matrix_fv;
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
  matrix_fv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, fv_2x3_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;
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
  matrix_fv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, fv_3x2_matrix_1stDeriv) {
  using stan::agrad::matrix_fv;
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
  matrix_fv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, fv_3x2_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;
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
  matrix_fv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffd_3x3_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;
  matrix_ffd Y(3,3);
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
  matrix_ffd output = stan::agrad::tcrossprod(Y);
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
}
TEST(AgradFwdMatrixTCrossProd, ffd_2x2_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;
  matrix_ffd Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_d X = Z * Z.transpose();
  matrix_ffd output = stan::agrad::tcrossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ(12,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ( 4,output(1,1).d_.val());
}
TEST(AgradFwdMatrixTCrossProd, ffd_1x1_matrix) {
  using stan::agrad::matrix_ffd;

  matrix_ffd Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffd output = stan::agrad::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());
}
TEST(AgradFwdMatrixTCrossProd, ffd_1x3_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;

  matrix_ffd Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_.val());
}
TEST(AgradFwdMatrixTCrossProd, ffd_2x3_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;

  matrix_ffd Y(2,3);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(1,2).d_ = 2.0;
  matrix_ffd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 14, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-20, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-20, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ( 98, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 24, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(-24, output(1,1).d_.val()); 
}
TEST(AgradFwdMatrixTCrossProd, ffd_3x2_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;

  matrix_ffd Y(3,2);
  Y << 1, 2,3,
    -1, 4, -9;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffd output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ( 5, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ( 1, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ( 1, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ(10, output(1,1).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(10, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(10, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ( 8, output(1,1).d_.val()); 
}

TEST(AgradFwdMatrixTCrossProd, ffv_3x3_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);
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
TEST(AgradFwdMatrixTCrossProd, ffv_3x3_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;

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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_3x3_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;

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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_3x3_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;

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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_2x2_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);
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
TEST(AgradFwdMatrixTCrossProd, ffv_2x2_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_2x2_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  matrix_ffv Y(2,2);
  Y << 3, 0,
     4, -3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_2x2_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x1_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(6,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x1_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x1_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x1_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,1);
  Y << 3;
   Y(0,0).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x3_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  EXPECT_FLOAT_EQ(14, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(24,output(0,0).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(6,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x3_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x3_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(0,2).d_ = 2.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(4,h[1]);
  EXPECT_FLOAT_EQ(4,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_1x3_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;

  matrix_ffv Y(1,3);
  Y << 1, 2,3;
   Y(0,0).d_ = 1.0;
   Y(0,1).d_ = 1.0;
   Y(0,2).d_ = 1.0;
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
  matrix_ffv output = stan::agrad::tcrossprod(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
}
TEST(AgradFwdMatrixTCrossProd, ffv_2x3_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_2x3_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_2x3_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_2x3_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_3x2_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
TEST(AgradFwdMatrixTCrossProd, ffv_3x2_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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

TEST(AgradFwdMatrixTCrossProd, ffv_3x2_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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

TEST(AgradFwdMatrixTCrossProd, ffv_3x2_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;
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
  matrix_ffv output = stan::agrad::tcrossprod(Y);

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
