#include <stan/math/fwd/mat/fun/determinant.hpp>
#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/mat/fun/multiply.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>

TEST(AgradMixMatrixDeterminant,matrix_fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(5.0,1.0);
  fvar<var> e(7.0,1.0);

  matrix_fv a(2,2);
  a << b,c,d,e;

  fvar<var> a_det = stan::math::determinant(a);

  EXPECT_FLOAT_EQ(-1,a_det.val_.val());
  EXPECT_FLOAT_EQ(1,a_det.d_.val());

  EXPECT_THROW(determinant(matrix_fv(2,3)), std::invalid_argument);

  AVEC z = createAVEC(b.val(),c.val(),d.val(),e.val());
  VEC h;
  a_det.val_.grad(z,h);
  EXPECT_FLOAT_EQ(7.0,h[0]);
  EXPECT_FLOAT_EQ(-5.0,h[1]);
  EXPECT_FLOAT_EQ(-3.0,h[2]);
  EXPECT_FLOAT_EQ(2.0,h[3]);
}
TEST(AgradMixMatrixDeterminant,matrix_fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> b(2.0,1.0);
  fvar<var> c(3.0,1.0);
  fvar<var> d(5.0,1.0);
  fvar<var> e(7.0,1.0);

  matrix_fv a(2,2);
  a << b,c,d,e;

  fvar<var> a_det = stan::math::determinant(a);

  AVEC z = createAVEC(b.val(),c.val(),d.val(),e.val());
  VEC h;
  a_det.d_.grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-1.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
}
TEST(AgradMixMatrixDeterminant,matrix_ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  fvar<fvar<var> > e(7.0,1.0);

  matrix_ffv a(2,2);
  a << b,c,d,e;

  fvar<fvar<var> > a_det = stan::math::determinant(a);

  EXPECT_FLOAT_EQ(-1,a_det.val_.val().val());
  EXPECT_FLOAT_EQ(1,a_det.d_.val().val());

  EXPECT_THROW(determinant(matrix_ffv(2,3)), std::invalid_argument);

  AVEC z = createAVEC(b.val().val(),c.val().val(),d.val().val(),e.val().val());
  VEC h;
  a_det.val_.val().grad(z,h);
  EXPECT_FLOAT_EQ(7.0,h[0]);
  EXPECT_FLOAT_EQ(-5.0,h[1]);
  EXPECT_FLOAT_EQ(-3.0,h[2]);
  EXPECT_FLOAT_EQ(2.0,h[3]);
}
TEST(AgradMixMatrixDeterminant,matrix_ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  fvar<fvar<var> > e(7.0,1.0);

  matrix_ffv a(2,2);
  a << b,c,d,e;

  fvar<fvar<var> > a_det = stan::math::determinant(a);

  AVEC z = createAVEC(b.val().val(),c.val().val(),d.val().val(),e.val().val());
  VEC h;
  a_det.val().d_.grad(z,h);
  EXPECT_FLOAT_EQ(0.0,h[0]);
  EXPECT_FLOAT_EQ(0.0,h[1]);
  EXPECT_FLOAT_EQ(0.0,h[2]);
  EXPECT_FLOAT_EQ(0.0,h[3]);
}

TEST(AgradMixMatrixDeterminant,matrix_ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  fvar<fvar<var> > e(7.0,1.0);

  matrix_ffv a(2,2);
  a << b,c,d,e;

  fvar<fvar<var> > a_det = stan::math::determinant(a);

  AVEC z = createAVEC(b.val().val(),c.val().val(),d.val().val(),e.val().val());
  VEC h;
  a_det.d_.val().grad(z,h);
  EXPECT_FLOAT_EQ(1.0,h[0]);
  EXPECT_FLOAT_EQ(-1.0,h[1]);
  EXPECT_FLOAT_EQ(-1.0,h[2]);
  EXPECT_FLOAT_EQ(1.0,h[3]);
}

TEST(AgradMixMatrixDeterminant,matrix_ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > b(2.0,1.0);
  fvar<fvar<var> > c(3.0,1.0);
  fvar<fvar<var> > d(5.0,1.0);
  fvar<fvar<var> > e(7.0,1.0);
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;

  matrix_ffv a(2,2);
  a << b,c,d,e;

  fvar<fvar<var> > a_det = stan::math::determinant(a);

  AVEC z = createAVEC(b.val().val(),c.val().val(),d.val().val(),e.val().val());
  VEC h;
  a_det.d_.d_.grad(z,h);
  EXPECT_NEAR(0.0,h[0],1e-8);
  EXPECT_NEAR(0.0,h[1],1e-8);
  EXPECT_NEAR(0.0,h[2],1e-8);
  EXPECT_NEAR(0.0,h[3],1e-8);
}
