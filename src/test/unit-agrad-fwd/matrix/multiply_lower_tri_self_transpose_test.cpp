#include <stan/agrad/fwd/matrix/multiply_lower_tri_self_transpose.hpp>
#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/multiply_lower_tri_self_transpose.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <test/unit/agrad/util.hpp>

using stan::agrad::fvar;
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fd_3x3_matrix) {
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
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fd output = stan::agrad::multiply_lower_tri_self_transpose(Y);
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
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fd_3x2_matrix) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  matrix_d Z(3,2);
  Z << 1, 0, 0,
    2, 3, 0;
  matrix_fd Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_fd output = stan::agrad::multiply_lower_tri_self_transpose(Y);
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
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fv_3x3_matrix_1stDeriv) {
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
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fv_3x3_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
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
  matrix_fv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(0,2).val(),Y(1,0).val(),Y(1,1).val(),Y(1,2).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fv_3x2_matrix_1stDeriv) {
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
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 2,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 2,output(1,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,2).d_.val());
  EXPECT_FLOAT_EQ( 8,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(2,2).d_.val());

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, fv_3x2_matrix_2ndDeriv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  matrix_fv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_fv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val(),Y(0,1).val(),Y(1,0).val(),Y(1,1).val());
  VEC h;
  output(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffd_3x3_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<double> > a,b,c,d,e,f,g;
  a.val_.val_ = 0.0;
  b.val_.val_ = 1.0;
  c.val_.val_ = 2.0;
  d.val_.val_ = 3.0;
  e.val_.val_ = 4.0;
  f.val_.val_ = 5.0;
  g.val_.val_ = 6.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;
  e.d_.val_ = 2.0;
  f.d_.val_ = 2.0;
  g.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << b,a,a,c,d,a,e,f,g;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_ffd output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val());
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffd_3x2_matrix) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  matrix_d Z(3,2);
  Z << 1, 0, 0,
    2, 3, 0;

  fvar<fvar<double> > a,b,c,d;
  a.val_.val_ = 0.0;
  b.val_.val_ = 1.0;
  c.val_.val_ = 2.0;
  d.val_.val_ = 3.0;
  a.d_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.d_.val_ = 2.0;
  d.d_.val_ = 2.0;

  matrix_ffd Y(3,2);
  Y << b,a,a,c,d,a;

  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_ffd output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 2,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 2,output(1,0).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,2).d_.val());
  EXPECT_FLOAT_EQ( 8,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(2,2).d_.val());
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x3_matrix_1stDeriv) {
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
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(20,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(60,output(2,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x3_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
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
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x3_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
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
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x3_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
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
   Y(0,0).val_.d_ = 1.0;
   Y(0,1).val_.d_ = 1.0;
   Y(0,2).val_.d_ = 1.0;
   Y(1,0).val_.d_ = 1.0;
   Y(1,1).val_.d_ = 1.0;
   Y(1,2).val_.d_ = 1.0;
   Y(2,0).val_.d_ = 1.0;
   Y(2,1).val_.d_ = 1.0;
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(0,2).val().val(),Y(1,0).val().val(),Y(1,1).val().val(),Y(1,2).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
  EXPECT_FLOAT_EQ(0,h[4]);
  EXPECT_FLOAT_EQ(0,h[5]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x2_matrix_1stDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  matrix_d Z(3,2);
  Z << 1, 0, 0,
    2, 3, 0;
  matrix_ffv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_d X = stan::math::multiply_lower_tri_self_transpose(Z);
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 2,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 2,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(2,2).d_.val().val());

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(2,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x2_matrix_2ndDeriv_1) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  matrix_ffv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x2_matrix_2ndDeriv_2) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  matrix_ffv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
   Y(0,0).d_ = 2.0;
   Y(0,1).d_ = 2.0;
   Y(1,0).d_ = 2.0;
   Y(1,1).d_ = 2.0;
   Y(2,0).d_ = 2.0;
   Y(2,1).d_ = 2.0;
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(4,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}

TEST(AgradFwdMatrixMultiplyLowerTriSelfTranspose, ffv_3x2_matrix_3rdDeriv) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  matrix_ffv Y(3,2);
  Y << 1, 0, 0,
    2, 3, 0;
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
  matrix_ffv output = stan::agrad::multiply_lower_tri_self_transpose(Y);

  AVEC q = createAVEC(Y(0,0).val().val(),Y(0,1).val().val(),Y(1,0).val().val(),Y(1,1).val().val());
  VEC h;
  output(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
