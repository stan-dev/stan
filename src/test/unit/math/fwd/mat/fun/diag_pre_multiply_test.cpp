#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/diag_pre_multiply.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradFwdMatrixDiagPreMultiply, vector_fd) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::vector_fd;
  using stan::math::vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fd Y(3,3);
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
  vector_fd B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A,Z);
  matrix_fd output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdMatrixDiagPreMultiply, vector_fd_exception) {
  using stan::math::matrix_fd;
  using stan::math::vector_fd;

  matrix_fd Y(3,3);
  matrix_fd Z(2,3);
  vector_fd B(3);
  vector_fd C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPreMultiply, rowvector_fd) {
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::row_vector_fd;
  using stan::math::row_vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fd Y(3,3);
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
  row_vector_fd B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A,Z);
  matrix_fd output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdMatrixDiagPreMultiply, rowvector_fd_exception) {
  using stan::math::matrix_fd;
  using stan::math::row_vector_fd;

  matrix_fd Y(3,3);
  matrix_fd Z(2,3);
  row_vector_fd B(3);
  row_vector_fd C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPreMultiply, vector_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::vector_ffd;
  using stan::math::vector_d;
  using stan::math::fvar;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_ffd B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A,Z);
  matrix_ffd output = stan::math::diag_pre_multiply(B,Y);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(16,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());
}
TEST(AgradFwdMatrixDiagPreMultiply, vector_ffd_exception) {
  using stan::math::matrix_ffd;
  using stan::math::vector_ffd;
  using stan::math::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  vector_ffd B(3);
  vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPreMultiply, rowvector_ffd) {
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffd;
  using stan::math::row_vector_d;
  using stan::math::fvar;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<double> > a;
  fvar<fvar<double> > b;
  fvar<fvar<double> > c;
  fvar<fvar<double> > d;
  fvar<fvar<double> > e;
  fvar<fvar<double> > f;
  a.val_.val_ = 1.0;
  a.d_.val_ = 2.0;  
  b.val_.val_ = 2.0;
  b.d_.val_ = 2.0;
  c.val_.val_ = 3.0;
  c.d_.val_ = 2.0;
  d.val_.val_ = 4.0;
  d.d_.val_ = 2.0;  
  e.val_.val_ = 5.0;
  e.d_.val_ = 2.0;
  f.val_.val_ = 6.0;
  f.d_.val_ = 2.0;

  matrix_ffd Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffd B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A,Z);
  matrix_ffd output = stan::math::diag_pre_multiply(B,Y);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val());
  EXPECT_FLOAT_EQ( 6,output(0,1).d_.val());
  EXPECT_FLOAT_EQ( 8,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(16,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());
}
TEST(AgradFwdMatrixDiagPreMultiply, rowvector_ffd_exception) {
  using stan::math::matrix_ffd;
  using stan::math::row_vector_ffd;
  using stan::math::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  row_vector_ffd B(3);
  row_vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
