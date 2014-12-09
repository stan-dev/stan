#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log1p_exp.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLog1pExp,Fvar) {
  using stan::agrad::fvar;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(1.0,2.0);
  fvar<double> z(2.0,3.0);

  fvar<double> a = log1p_exp(x);
  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), a.d_);

  fvar<double> b = log1p_exp(y);
  EXPECT_FLOAT_EQ(log1p_exp(1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.0) / (1 + exp(1.0)), b.d_);

  fvar<double> a2 = log(1+exp(x));
  EXPECT_FLOAT_EQ(a.d_, a2.d_);

  fvar<double> b2 = log(1+exp(y));
  EXPECT_FLOAT_EQ(b.d_, b2.d_);
}

TEST(AgradFwdLog1pExp,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1p_exp(x);

  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(0.5) / (1 + exp(0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), g[0]);
}
TEST(AgradFwdLog1pExp,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1p_exp(x);

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), g[0]);
}
TEST(AgradFwdLog1pExp,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log1p_exp(x);

  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log1p_exp(y);
  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLog1pExp,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p_exp(x);

  EXPECT_FLOAT_EQ(log1p_exp(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p_exp(y);
  EXPECT_FLOAT_EQ(log1p_exp(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)), r[0]);
}
TEST(AgradFwdLog1pExp,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1p_exp;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log1p_exp(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log1p_exp(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(0.5) / (1 + exp(0.5)) / (1 + exp(0.5)), r[0]);
}
TEST(AgradFwdLog1pExp,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log1p_exp(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(-0.05755679485232, g[0]);
}

struct log1p_exp_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log1p_exp(arg1);
  }
};

TEST(AgradFwdLog1pExp,log1p_exp_NaN) {
  log1p_exp_fun log1p_exp_;
  test_nan(log1p_exp_,false);
}
