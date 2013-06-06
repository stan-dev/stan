#include <gtest/gtest.h>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/diag_pre_multiply.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFwdMatrix, diag_pre_multiply_vector) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::vector_fv;
  using stan::math::vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fv Y(3,3);
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
  vector_fv B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdMatrix, diag_pre_multiply_vector_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  vector_fv B(3);
  vector_fv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdMatrix, diag_pre_multiply_rowvector) {
  using stan::agrad::matrix_fv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fv;
  using stan::math::row_vector_d;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;
  matrix_fv Y(3,3);
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
  row_vector_fv B(3);
  B << 1, 2, 3;
   B(0).d_ = 2.0;
   B(1).d_ = 2.0;
   B(2).d_ = 2.0;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdMatrix, diag_pre_multiply_rowvector_exception) {
  using stan::agrad::matrix_fv;
  using stan::agrad::row_vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  row_vector_fv B(3);
  row_vector_fv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix, diag_pre_multiply_vector) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fvv;
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

  matrix_fvv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_fvv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fvv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdFvarVarMatrix, diag_pre_multiply_vector_exception) {
  using stan::agrad::matrix_fvv;
  using stan::agrad::vector_fvv;

  matrix_fvv Y(3,3);
  matrix_fvv Z(2,3);
  vector_fvv B(3);
  vector_fvv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdFvarVarMatrix, diag_pre_multiply_rowvector) {
  using stan::agrad::matrix_fvv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_fvv;
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

  matrix_fvv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_fvv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_fvv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdFvarVarMatrix, diag_pre_multiply_rowvector_exception) {
  using stan::agrad::matrix_fvv;
  using stan::agrad::row_vector_fvv;

  matrix_fvv Y(3,3);
  matrix_fvv Z(2,3);
  row_vector_fvv B(3);
  row_vector_fvv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix, diag_pre_multiply_vector) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

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

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_ffv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdFvarFvarMatrix, diag_pre_multiply_vector_exception) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  vector_ffv B(3);
  vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
TEST(AgradFwdFvarFvarMatrix, diag_pre_multiply_rowvector) {
  using stan::agrad::matrix_ffv;
  using stan::math::matrix_d;
  using stan::agrad::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::agrad::fvar;
  using stan::agrad::var;

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

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_pre_multiply(A, Z);
  matrix_ffv output = stan::math::diag_pre_multiply(B,Y);

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
TEST(AgradFwdFvarFvarMatrix, diag_pre_multiply_rowvector_exception) {
  using stan::agrad::matrix_ffv;
  using stan::agrad::row_vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  row_vector_ffv B(3);
  row_vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_pre_multiply(Y,B), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_pre_multiply(Y,Z), std::domain_error);
}
