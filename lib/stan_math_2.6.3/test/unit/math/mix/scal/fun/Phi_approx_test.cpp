// Phi_approx needs inv_logit in order for this to work
#include <gtest/gtest.h>
#include <stan/math/prim/scal/fun/Phi_approx.hpp>
#include <stan/math/rev/scal/fun/Phi_approx.hpp>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/mix/scal/fun/nan_util.hpp>
#include <stan/math/fwd/scal/fun/pow.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/scal/fun/exp.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/pow.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/fwd/scal/fun/inv_logit.hpp>
#include <stan/math/rev/scal/fun/inv_logit.hpp>

TEST(AgradFwdPhi_approx, FvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::var;
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
  using stan::math::fvar;
  using stan::math::var;
  using stan::math::Phi_approx;

  fvar<var> x(1.0,1.3);
  fvar<var> a = Phi_approx(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.3143405, g[0]);
}

TEST(AgradFwdPhi_approx, FvarFvarVar_1stDeriv) {
  using stan::math::fvar;
  using stan::math::Phi_approx;
  using stan::math::var;

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
  using stan::math::fvar;
  using stan::math::Phi_approx;
  using stan::math::var;

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
  using stan::math::fvar;
  using stan::math::Phi_approx;
  using stan::math::var;

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
  test_nan_mix(Phi_approx_,false);
}
