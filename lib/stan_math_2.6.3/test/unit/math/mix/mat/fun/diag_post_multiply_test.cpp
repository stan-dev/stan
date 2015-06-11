#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/diag_post_multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>

using stan::math::matrix_d;
using stan::math::vector_d;
using stan::math::row_vector_d;
using stan::math::diag_post_multiply;

TEST(AgradMixMatrixDiagPostMultiply, vector_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::vector_fv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

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
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradMixMatrixDiagPostMultiply, vector_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::vector_fv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_fv B(3);
  B << a,b,c;

  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, vector_fv_exception) {
  using stan::math::matrix_fv;
  using stan::math::vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  vector_fv B(3);
  vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradMixMatrixDiagPostMultiply, rowvector_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

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
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::row_vector_fv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);

  matrix_fv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_fv B(3);
  B << a,b,c;

  matrix_fv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_fv_exception) {
  using stan::math::matrix_fv;
  using stan::math::row_vector_fv;

  matrix_fv Y(3,3);
  matrix_fv Z(2,3);
  row_vector_fv B(3);
  row_vector_fv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(C,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
TEST(AgradMixMatrixDiagPostMultiply, vector_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::vector_ffv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  vector_d A(3);
  A << 1, 2, 3;
  vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, vector_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::vector_ffv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradMixMatrixDiagPostMultiply, vector_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::vector_ffv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, vector_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::vector_ffv;
  using stan::math::vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(5.0,1.0);
  fvar<fvar<var> > f(6.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, vector_ffv_exception) {
  using stan::math::matrix_ffv;
  using stan::math::vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  vector_ffv B(3);
  vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}

TEST(AgradMixMatrixDiagPostMultiply, rowvector_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 2, 3,
    2, 3, 4,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;

  row_vector_d A(3);
  A << 1, 2, 3;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_d W = stan::math::diag_post_multiply(Z,A);
  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(W(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 4,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ( 8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(12,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ( 6,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(10,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(14,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(18,output(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::row_vector_ffv;
  using stan::math::row_vector_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(5.0,1.0);
  fvar<fvar<var> > f(6.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(3,3);
  Y << a,b,c,b,c,d,d,e,f;
  row_vector_ffv B(3);
  B << a,b,c;

  matrix_ffv output = stan::math::diag_post_multiply(Y,B);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixDiagPostMultiply, rowvector_ffv_exception) {
  using stan::math::matrix_ffv;
  using stan::math::row_vector_ffv;

  matrix_ffv Y(3,3);
  matrix_ffv Z(2,3);
  row_vector_ffv B(3);
  row_vector_ffv C(4);

  EXPECT_THROW(stan::math::diag_post_multiply(B,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,C), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Z,Y), std::domain_error);
  EXPECT_THROW(stan::math::diag_post_multiply(Y,Z), std::domain_error);
}
