#include <gtest/gtest.h>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/mix/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/mat/fun/LDLT_factor.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/fwd/scal/fun/abs.hpp>
#include <stan/math/rev/scal/fun/abs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log.hpp>

TEST(AgradMixMatrixLogDeterminantLDLT,fv_1stDeriv) {
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<var> a(3.0,1.0);
  fvar<var> b(0.0,2.0);
  fvar<var> c(0.0,2.0);
  fvar<var> d(4.0,2.0);

  matrix_fv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_v;
  ldlt_v.compute(v);

  fvar<var> det;
  det = log_determinant_ldlt(ldlt_v);
  EXPECT_FLOAT_EQ(std::log(12.0), det.val_.val());
  EXPECT_FLOAT_EQ(0.83333333, det.d_.val());

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  det.val_.grad(q,h);
  EXPECT_FLOAT_EQ(0.33333333,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0.25,h[3]);
}
TEST(AgradMixMatrixLogDeterminantLDLT,fv_2ndDeriv) {
  using stan::math::matrix_fv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<var> a(3.0,1.0);
  fvar<var> b(0.0,2.0);
  fvar<var> c(0.0,2.0);
  fvar<var> d(4.0,2.0);
  matrix_fv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<var>,-1,-1> ldlt_v;
  ldlt_v.compute(v);

  fvar<var> det;
  det = log_determinant_ldlt(ldlt_v);

  AVEC q = createAVEC(a.val(),b.val(),c.val(),d.val());
  VEC h;
  det.d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.11111111,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(-0.33333333,h[2]);
  EXPECT_FLOAT_EQ(-0.125,h[3]);
}
TEST(AgradMixMatrixLogDeterminantLDLT,ffv_1stDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<fvar<var> > a(3.0,1.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(0.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);

  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_v;
  ldlt_v.compute(v);

  fvar<fvar<var> > det;
  det = log_determinant_ldlt(ldlt_v);
  EXPECT_FLOAT_EQ(std::log(12.0), det.val_.val().val());
  EXPECT_FLOAT_EQ(0.83333333, det.d_.val().val());

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.val_.val().grad(q,h);
  EXPECT_FLOAT_EQ(0.33333333,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0,h[2]);
  EXPECT_FLOAT_EQ(0.25,h[3]);
}
TEST(AgradMixMatrixLogDeterminantLDLT,ffv_2ndDeriv_1) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<fvar<var> > a(3.0,1.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(0.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;
  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_v;
  ldlt_v.compute(v);
  
  fvar<fvar<var> > det;
  det = log_determinant_ldlt(ldlt_v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.val().d_.grad(q,h);
  EXPECT_FLOAT_EQ(-0.11111111,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(-0.16666667,h[2]);
  EXPECT_FLOAT_EQ(-0.0625,h[3]);
}
TEST(AgradMixMatrixLogDeterminantLDLT,ffv_2ndDeriv_2) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<fvar<var> > a(3.0,1.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(0.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_v;
  ldlt_v.compute(v);

  fvar<fvar<var> > det;
  det = log_determinant_ldlt(ldlt_v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.d_.val().grad(q,h);
  EXPECT_FLOAT_EQ(-0.11111111,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(-0.33333333,h[2]);
  EXPECT_FLOAT_EQ(-.125,h[3]);
}
TEST(AgradMixMatrixLogDeterminantLDLT,ffv_3rdDeriv) {
  using stan::math::matrix_ffv;
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::log_determinant_ldlt;
  
  fvar<fvar<var> > a(3.0,1.0);
  fvar<fvar<var> > b(0.0,2.0);
  fvar<fvar<var> > c(0.0,2.0);
  fvar<fvar<var> > d(4.0,2.0);
  a.val_.d_ = 1.0;
  b.val_.d_ = 1.0;
  c.val_.d_ = 1.0;
  d.val_.d_ = 1.0;

  matrix_ffv v(2,2);
  v << a,b,c,d;
  
  stan::math::LDLT_factor<fvar<fvar<var> >,-1,-1> ldlt_v;
  ldlt_v.compute(v);

  fvar<fvar<var> > det;
  det = log_determinant_ldlt(ldlt_v);

  AVEC q = createAVEC(a.val().val(),b.val().val(),c.val().val(),d.val().val());
  VEC h;
  det.d_.d_.grad(q,h);
  EXPECT_FLOAT_EQ(0.18518518,h[0]);
  EXPECT_FLOAT_EQ(0,h[1]);
  EXPECT_FLOAT_EQ(0.33333333,h[2]);
  EXPECT_FLOAT_EQ(0.14583333,h[3]);
}
