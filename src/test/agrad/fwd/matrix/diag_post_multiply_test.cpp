#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/diag_post_multiply.hpp>
#include <stan/agrad/var.hpp>

TEST(AgradFwdMatrixDiagPostMultiply, vector_fd) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  using stan::agrad::vector_fd;
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

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fd output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, vector_fd_exception) {
  using stan::agrad::matrix_fd;
  using stan::agrad::vector_fd;

  matrix_fd Y(3,3);
  matrix_fd Z(2,3);
  vector_fd B(3);
  vector_fd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fd) {
  using stan::agrad::matrix_fd;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fd;
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

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fd output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fd_exception) {
  using stan::agrad::matrix_fd;
  using stan::agrad::row_vector_fd;

  matrix_fd Y(3,3);
  matrix_fd Z(2,3);
  row_vector_fd B(3);
  row_vector_fd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_fv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_fv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fv output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, vector_fv_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  vector_fv B(3);
  vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fv) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_fv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_fv output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_fv_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;
  using stan::agrad::fvar;
  using stan::agrad::var;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  row_vector_fv B(3);
  row_vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::vector_ffd;
  using stan::math::vector_d;
  using stan::agrad::fvar;

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

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffd output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, vector_ffd_exception) {
  using stan::agrad::matrix_ffd;
  using stan::agrad::vector_ffd;
  using stan::agrad::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  vector_ffd B(3);
  vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffd) {
  using stan::agrad::matrix_ffd;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffd;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;

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

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffd output = stan::math::diag_post_multiply(Y,B);

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
TEST(AgradFwdMatrixDiagPostMultiply, rowvector_ffd_exception) {
  using stan::agrad::matrix_ffd;
  using stan::agrad::row_vector_ffd;
  using stan::agrad::fvar;

  matrix_ffd Y(3,3);
  matrix_ffd Z(2,3);
  row_vector_ffd B(3);
  row_vector_ffd C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
