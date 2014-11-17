#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/Phi_approx.hpp>
#include <stan/prob/distributions/univariate/continuous/normal.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdPhi_approx,Fvar) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;
  fvar<double> x = 1.0;
  x.d_ = 1.0;
  
  fvar<double> Phi_approx_x = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), Phi_approx_x.val_);
  EXPECT_FLOAT_EQ(0.24152729,Phi_approx_x.d_);
}
TEST(AgradFwdPhi_approx, FvarDerivUnderOverFlow) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;

  fvar<double> x = -27.5;
  x.d_ = 1.0;
  fvar<double> Phi_approx_x = Phi_approx(x);
  EXPECT_FLOAT_EQ(0, Phi_approx_x.d_);

  fvar<double> y = 27.5;
  y.d_ = 1.0;
  fvar<double> Phi_approx_y = Phi_approx(y);
  EXPECT_FLOAT_EQ(0, Phi_approx_y.d_);
}
TEST(AgradFwdPhi_approx, FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::Phi_approx;

  fvar<var> x(1.0,1.3);
  fvar<var> a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val());
  EXPECT_FLOAT_EQ(0.31398547,a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0.24152729, g[0]);
}
TEST(AgradFwdPhi_approx, FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::Phi_approx;

  fvar<var> x(1.0,1.3);
  fvar<var> a = Phi_approx(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.3143405, g[0]);
}
TEST(AgradFwdPhi_approx, FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;

  fvar<fvar<double> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0.24152729, a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  a = Phi_approx(y);
  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(0.24152729, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdPhi_approx, FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  EXPECT_FLOAT_EQ(Phi_approx(1.0), a.val_.val_.val());
  EXPECT_FLOAT_EQ(0.24152729,a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(0.24152729, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi_approx(y);
  EXPECT_FLOAT_EQ(0.84133035, b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(0.24152729,b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(0.24152729, r[0]);
}

TEST(AgradFwdPhi_approx, FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 1.0;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.24180038, g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 1.0;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = Phi_approx(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(-0.24180038, r[0]);
}
TEST(AgradFwdPhi_approx, FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::math::Phi_approx;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 2.0;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = Phi_approx(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.1590839, g[0]);
}

struct Phi_approx_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return stan::math::Phi_approx(arg1);
  }
};

TEST(AgradFwdPhi_approx,Phi_approx_NaN) {
  Phi_approx_fun Phi_approx_;
  test_nan(Phi_approx_,false);
}
