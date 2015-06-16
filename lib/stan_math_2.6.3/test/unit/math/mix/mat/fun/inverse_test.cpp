#include <stan/math/rev/mat/fun/multiply.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>

TEST(AgradMixMatrixInverse,fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> d(2.0,1.0);
  fvar<var> e(3.0,1.0);
  fvar<var> f(5.0,1.0);
  fvar<var> g(7.0,1.0);

  matrix_fv a(2,2);
  a << d,e,f,g;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_fv a_inv = stan::math::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val(),1.0E-12);

  EXPECT_THROW(stan::math::inverse(matrix_fv(2,3)), std::invalid_argument);

  AVEC q = createAVEC(d.val(),e.val(),f.val(),g.val());
  VEC h;
  a_inv(0,0).val_.grad(q,h);
  EXPECT_FLOAT_EQ(-49.0,h[0]);
  EXPECT_FLOAT_EQ(35.0,h[1]);
  EXPECT_FLOAT_EQ(21.0,h[2]);
  EXPECT_FLOAT_EQ(-15.0,h[3]);
}
TEST(AgradMixMatrixInverse,fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<var> d(2.0,1.0);
  fvar<var> e(3.0,1.0);
  fvar<var> f(5.0,1.0);
  fvar<var> g(7.0,1.0);

  matrix_fv a(2,2);
  a << d,e,f,g;

  matrix_fv a_inv = stan::math::inverse(a);

  AVEC q = createAVEC(d.val(),e.val(),f.val(),g.val());
  VEC h;
  a_inv(0,0).d_.grad(q,h);
  EXPECT_FLOAT_EQ(-112.0,h[0]);
  EXPECT_FLOAT_EQ(82.0,h[1]);
  EXPECT_FLOAT_EQ(52.0,h[2]);
  EXPECT_FLOAT_EQ(-38.0,h[3]);
}
TEST(AgradMixMatrixInverse,ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

   matrix_d b(2,2);
   b << 2.0, 3.0, 5.0,7.0;
   b = b.inverse();

  matrix_ffv a_inv = stan::math::inverse(a);

  EXPECT_NEAR(b(0,0),a_inv(0,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(0,1),a_inv(0,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(1,0),a_inv(1,0).val_.val().val(),1.0E-12);
  EXPECT_NEAR(b(1,1),a_inv(1,1).val_.val().val(),1.0E-12);
  EXPECT_NEAR(-8,a_inv(0,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR( 4,a_inv(0,1).d_.val().val(),1.0E-12);
  EXPECT_NEAR( 6,a_inv(1,0).d_.val().val(),1.0E-12);
  EXPECT_NEAR(-3,a_inv(1,1).d_.val().val(),1.0E-12);

  EXPECT_THROW(stan::math::inverse(matrix_ffv(2,3)), std::invalid_argument);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-49.0,h[0]);
  EXPECT_FLOAT_EQ(35.0,h[1]);
  EXPECT_FLOAT_EQ(21.0,h[2]);
  EXPECT_FLOAT_EQ(-15.0,h[3]);
}
TEST(AgradMixMatrixInverse,ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::math::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(0,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0,h[3]);
}
TEST(AgradMixMatrixInverse,ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::math::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-112.0,h[0]);
  EXPECT_FLOAT_EQ(82.0,h[1]);
  EXPECT_FLOAT_EQ(52.0,h[2]);
  EXPECT_FLOAT_EQ(-38.0,h[3]);
}
TEST(AgradMixMatrixInverse,ffv_3rDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::matrix_d;
  using stan::math::fvar;
  using stan::math::var;

  fvar<fvar<var> > d(2.0,1.0);
  fvar<fvar<var> > e(3.0,1.0);
  fvar<fvar<var> > f(5.0,1.0);
  fvar<fvar<var> > g(7.0,1.0);
  d.val_.d_ = 1.0;
  e.val_.d_ = 1.0;
  f.val_.d_ = 1.0;
  g.val_.d_ = 1.0;

  matrix_ffv a(2,2);
  a << d,e,f,g;

  matrix_ffv a_inv = stan::math::inverse(a);

  AVEC q = createAVEC(d.val().val(),e.val().val(),f.val().val(),g.val().val());
  VEC h;
  a_inv(0,0).d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-352,h[0]);
  EXPECT_FLOAT_EQ(260,h[1]);
  EXPECT_FLOAT_EQ(168,h[2]);
  EXPECT_FLOAT_EQ(-124,h[3]);
}
