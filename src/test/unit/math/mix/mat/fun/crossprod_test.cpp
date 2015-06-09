#include <stan/math/fwd/mat/fun/crossprod.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>

TEST(AgradMixMatrixCrossProd, 3x3_matrix_fv_1stderiv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);
  fvar<var> g(0.0,2.0);
  matrix_fv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_d X = Z.transpose() * Z;
  matrix_fv output = stan::math::crossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ(28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(30,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(26,output(0,2).d_.val());
  EXPECT_FLOAT_EQ(30,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(32,output(1,1).d_.val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val());
  EXPECT_FLOAT_EQ(26,output(2,0).d_.val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val());
  EXPECT_FLOAT_EQ(24,output(2,2).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val(),g.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(4.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(8.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 3x3_matrix_fv_2ndderiv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(4.0,2.0);
  fvar<var> e(5.0,2.0);
  fvar<var> f(6.0,2.0);
  fvar<var> g(0.0,2.0);
  matrix_fv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val(),g.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(4.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(4.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<var> a(3.0,2.0);
  fvar<var> b(0.0,2.0);
  fvar<var> c(4.0,2.0);
  fvar<var> d(-3.0,2.0);

  matrix_fv Y(2,2);
  Y << a,b,c,d;
  matrix_d X = Z.transpose() * Z;
  matrix_fv output = stan::math::crossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val());
  }

  EXPECT_FLOAT_EQ( 28,output(0,0).d_.val());
  EXPECT_FLOAT_EQ(  8,output(0,1).d_.val());
  EXPECT_FLOAT_EQ(  8,output(1,0).d_.val());
  EXPECT_FLOAT_EQ(-12,output(1,1).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(6.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<var> a(3.0,2.0);
  fvar<var> b(0.0,2.0);
  fvar<var> c(4.0,2.0);
  fvar<var> d(-3.0,2.0);

  matrix_fv Y(2,2);
  Y << a,b,c,d;
  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(3.0,2.0);

  matrix_fv Y(1,1);
  Y << a;
  matrix_fv output = stan::math::crossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val());

  AVEC z = createAVEC(a.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(6.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(3.0,2.0);

  matrix_fv Y(1,1);
  Y << a;
  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);

  matrix_fv Y(1,3);
  Y << a,b,c;
  matrix_fv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ(1, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(4,output(0,0).d_.val());

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);

  matrix_fv Y(1,3);
  Y << a,b,c;
  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val(),b.val(),c.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 2x3_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_fv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ( 2, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-2, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ(20, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 0, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(12, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(24, output(1,1).d_.val()); 

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-2.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 2x3_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(4.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 3x2_matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_fv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ( 26, output(0,0).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(0,1).val_.val()); 
  EXPECT_FLOAT_EQ(-37, output(1,0).val_.val()); 
  EXPECT_FLOAT_EQ( 86, output(1,1).val_.val());
  EXPECT_FLOAT_EQ( 32, output(0,0).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val()); 
  EXPECT_FLOAT_EQ(-32, output(1,1).d_.val()); 

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).val_.grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(6.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(8.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 3x2_matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> a(1.0,2.0);
  fvar<var> b(2.0,2.0);
  fvar<var> c(3.0,2.0);
  fvar<var> d(-1.0,2.0);
  fvar<var> e(4.0,2.0);
  fvar<var> f(-9.0,2.0);

  matrix_fv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_fv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val(),b.val(),c.val(),d.val(),e.val(),f.val());
  VEC h;
  output(0,0).d_.grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(4.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}

TEST(AgradMixMatrixCrossProd, 3x3_matrix_ffv_1stderiv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);
  fvar<fvar<var> > g(0.0,2.0);
  matrix_ffv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_d X = Z.transpose() * Z;
  matrix_ffv output = stan::math::crossprod(Y);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ(28,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(30,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(26,output(0,2).d_.val().val());
  EXPECT_FLOAT_EQ(30,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(32,output(1,1).d_.val().val());
  EXPECT_FLOAT_EQ(28,output(1,2).d_.val().val());
  EXPECT_FLOAT_EQ(26,output(2,0).d_.val().val());
  EXPECT_FLOAT_EQ(28,output(2,1).d_.val().val());
  EXPECT_FLOAT_EQ(24,output(2,2).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(4.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(8.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 3x3_matrix_ffv_2ndderiv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);
  fvar<fvar<var> > g(0.0,2.0);
  matrix_ffv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  output(0,0).val_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 3x3_matrix_ffv_2ndderiv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  fvar<fvar<var> > e(5.0,2.0);
  fvar<fvar<var> > f(6.0,2.0);
  fvar<fvar<var> > g(0.0,2.0);
  matrix_ffv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(4.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(4.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 3x3_matrix_ffv_3rdderiv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(3,3);
  Z << 1, 0, 0,
    2, 3, 0,
    4, 5, 6;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(4.0,1.0);
  fvar<fvar<var> > e(5.0,1.0);
  fvar<fvar<var> > f(6.0,1.0);
  fvar<fvar<var> > g(0.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;
  g.val_.d_ = 1.0;
  matrix_ffv Y(3,3);
  Y << a,g,g,b,c,g,d,e,f;

  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
  EXPECT_FLOAT_EQ(0.0,h[6]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<fvar<var> > a(3.0,2.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(4.0,2.0);
  fvar<fvar<var> > d(-3.0,2.0);

  matrix_ffv Y(2,2);
  Y << a,b,c,d;
  matrix_d X = Z.transpose() * Z;
  matrix_ffv output = stan::math::crossprod(Y);
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(X(i,j), output(i,j).val_.val().val());
  }

  EXPECT_FLOAT_EQ( 28,output(0,0).d_.val().val());
  EXPECT_FLOAT_EQ(  8,output(0,1).d_.val().val());
  EXPECT_FLOAT_EQ(  8,output(1,0).d_.val().val());
  EXPECT_FLOAT_EQ(-12,output(1,1).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(6.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(8.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<fvar<var> > a(3.0,2.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(4.0,2.0);
  fvar<fvar<var> > d(-3.0,2.0);

  matrix_ffv Y(2,2);
  Y << a,b,c,d;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<fvar<var> > a(3.0,2.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(4.0,2.0);
  fvar<fvar<var> > d(-3.0,2.0);

  matrix_ffv Y(2,2);
  Y << a,b,c,d;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 2x2_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  matrix_d Z(2,2);
  Z <<3, 0,
     4, -3;

  fvar<fvar<var> > a(3.0,1.0);
  fvar<fvar<var> > b(0.0,1.0);
  fvar<fvar<var> > c(4.0,1.0);
  fvar<fvar<var> > d(-3.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  matrix_ffv Y(2,2);
  Y << a,b,c,d;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(3.0,2.0);

  matrix_ffv Y(1,1);
  Y << a;
  matrix_ffv output = stan::math::crossprod(Y);
  EXPECT_FLOAT_EQ( 9, output(0,0).val_.val().val());
  EXPECT_FLOAT_EQ(12, output(0,0).d_.val().val());

  AVEC z = createAVEC(a.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(6.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(3.0,2.0);

  matrix_ffv Y(1,1);
  Y << a;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(3.0,2.0);

  matrix_ffv Y(1,1);
  Y << a;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x1_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(3.0,1.0);
  a.val_.d_ = 1.0;

  matrix_ffv Y(1,1);
  Y << a;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);

  matrix_ffv Y(1,3);
  Y << a,b,c;
  matrix_ffv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ(1, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(4,output(0,0).d_.val().val());

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);

  matrix_ffv Y(1,3);
  Y << a,b,c;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  output(0,0).val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);

  matrix_ffv Y(1,3);
  Y << a,b,c;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 1x3_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;

  matrix_ffv Y(1,3);
  Y << a,b,c;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val());
  VEC h;
  output(0,0).d_.d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
}
TEST(AgradMixMatrixCrossProd, 2x3_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ( 2, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(-2, output(0,1).val_.val().val()); 
  EXPECT_FLOAT_EQ(-2, output(1,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(20, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( 0, output(0,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(12, output(0,1).d_.val().val()); 
  EXPECT_FLOAT_EQ(12, output(1,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(24, output(1,1).d_.val().val()); 

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(-2.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 2x3_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

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
TEST(AgradMixMatrixCrossProd, 2x3_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(4.0,h[3]);
  EXPECT_FLOAT_EQ(0.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 2x3_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(-1.0,1.0);
  fvar<fvar<var> > e(4.0,1.0);
  fvar<fvar<var> > f(-9.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(2,3);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

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
TEST(AgradMixMatrixCrossProd, 3x2_matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

  EXPECT_FLOAT_EQ( 26, output(0,0).val_.val().val()); 
  EXPECT_FLOAT_EQ(-37, output(0,1).val_.val().val()); 
  EXPECT_FLOAT_EQ(-37, output(1,0).val_.val().val()); 
  EXPECT_FLOAT_EQ( 86, output(1,1).val_.val().val());
  EXPECT_FLOAT_EQ( 32, output(0,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(  0, output(0,1).d_.val().val()); 
  EXPECT_FLOAT_EQ(  0, output(1,0).d_.val().val()); 
  EXPECT_FLOAT_EQ(-32, output(1,1).d_.val().val()); 

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(2.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(6.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(8.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 3x2_matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

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
TEST(AgradMixMatrixCrossProd, 3x2_matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,2.0);
  fvar<fvar<var> > b(2.0,2.0);
  fvar<fvar<var> > c(3.0,2.0);
  fvar<fvar<var> > d(-1.0,2.0);
  fvar<fvar<var> > e(4.0,2.0);
  fvar<fvar<var> > f(-9.0,2.0);

  matrix_ffv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

  AVEC z = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val(),e.val().val(),f.val().val());
  VEC h;
  output(0,0).d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(4.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(4.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
  EXPECT_FLOAT_EQ(4.0,h[4]);
  EXPECT_FLOAT_EQ(0.0,h[5]);
}
TEST(AgradMixMatrixCrossProd, 3x2_matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > a(1.0,1.0);
  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(-1.0,1.0);
  fvar<fvar<var> > e(4.0,1.0);
  fvar<fvar<var> > f(-9.0,1.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;

  matrix_ffv Y(3,2);
  Y << a,b,c,d,e,f;
  matrix_ffv output = stan::math::crossprod(Y);

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
