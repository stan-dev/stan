#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/inv_sqrt.hpp>
#include <stan/math/constants.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdInvSqrt,Fvar) {
  using stan::agrad::fvar;
  using stan::math::inv_sqrt;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // Derivatives w.r.t. x
  fvar<double> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(-0.5 / (0.5 * std::sqrt(0.5)), a.d_);

  fvar<double> y(0.0);
  y.d_ = 1.0;
  fvar<double> g = inv_sqrt(y);

  EXPECT_FLOAT_EQ(stan::math::positive_infinity(), g.val_);
  EXPECT_FLOAT_EQ(stan::math::negative_infinity(), g.d_);

  fvar<double> z(-1.0);
  z.d_ = 2.0;
  g = inv_sqrt(z);

  std::isnan(g.val_);
  std::isnan(g.d_);
}   

TEST(AgradFwdInvSqrt,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_sqrt;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 / (0.5 * std::sqrt(0.5)), g[0]);
}
TEST(AgradFwdInvSqrt,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_sqrt;

  fvar<var> x(0.5,1.0);
  fvar<var> a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(-0.5 * -1.5 / (0.25 * std::sqrt(0.5)), g[0]);
}
TEST(AgradFwdInvSqrt,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::inv_sqrt;
  using std::log;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdInvSqrt,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), g[0]);
}
TEST(AgradFwdInvSqrt,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  EXPECT_FLOAT_EQ(inv_sqrt(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(-0.5 * inv_sqrt(0.5) / (0.5), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.5 * -1.5 / (0.25 * std::sqrt(0.5)), g[0]);
}
TEST(AgradFwdInvSqrt,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_sqrt;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = inv_sqrt(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-21.2132034355964257320253308631, g[0]);
}

struct inv_sqrt_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return inv_sqrt(arg1);
  }
};

TEST(AgradFwdInvSqrt,inv_sqrt_NaN) {
  inv_sqrt_fun inv_sqrt_;
  test_nan(inv_sqrt_,false);
}
