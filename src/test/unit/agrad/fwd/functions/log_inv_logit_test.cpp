#include <gtest/gtest.h>
#include <stan/agrad/fwd.hpp>
#include <stan/math/functions/log_inv_logit.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit/agrad/util.hpp>
#include <test/unit/agrad/fwd/nan_util.hpp>

TEST(AgradFwdLogInvLogit,Fvar) {
  using stan::agrad::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<double> x(0.5,1.0);
  fvar<double> y(-1.0,2.0);
  fvar<double> z(0.0,3.0);

  fvar<double> a = log_inv_logit(x);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1.0 * exp(-0.5) / (1 + exp(-0.5)), a.d_);

  fvar<double> b = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(-1.0), b.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.0) / (1 + exp(1.0)), b.d_);

  fvar<double> c = log_inv_logit(z);
  EXPECT_FLOAT_EQ(log_inv_logit(0.0), c.val_);
  EXPECT_FLOAT_EQ(3.0 * exp(0.0) / (1 + exp(0.0)), c.d_);
}

TEST(AgradFwdLogInvLogit,FvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(-0.5) / (1 + exp(-0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), g[0]);
}
TEST(AgradFwdLogInvLogit,FvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(-0.5) / (1 + exp(-0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.3 * (-exp(-0.5) * (1 + exp(-0.5)) + exp(-0.5) * exp(-0.5)) 
                  / (1 + exp(-0.5)) / (1 + exp(-0.5)), g[0]);
}
TEST(AgradFwdLogInvLogit,FvarFvarDouble) {
  using stan::agrad::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
TEST(AgradFwdLogInvLogit,FvarFvarVar_1stDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_.val());
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.val_.d_.val());
  EXPECT_FLOAT_EQ(0, a.d_.val_.val());
  EXPECT_FLOAT_EQ(0, a.d_.d_.val());

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.val_.grad(p,g);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), b.val_.val_.val());
  EXPECT_FLOAT_EQ(0, b.val_.d_.val());
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), b.d_.val_.val());
  EXPECT_FLOAT_EQ(0, b.d_.d_.val());

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.val_.val_.grad(q,r);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), r[0]);
}
TEST(AgradFwdLogInvLogit,FvarFvarVar_2ndDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<var> > a = log_inv_logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.val_.d_.grad(p,g);
  EXPECT_FLOAT_EQ((-exp(-0.5) * (1 + exp(-0.5)) + exp(-0.5) * exp(-0.5))
                  / (1 + exp(-0.5)) / (1 + exp(-0.5)), g[0]);

  fvar<fvar<var> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  fvar<fvar<var> > b = log_inv_logit(y);

  AVEC q = createAVEC(y.val_.val_);
  VEC r;
  b.d_.val_.grad(q,r);
  EXPECT_FLOAT_EQ((-exp(-0.5) * (1 + exp(-0.5)) + exp(-0.5) * exp(-0.5)) 
                  / (1 + exp(-0.5)) / (1 + exp(-0.5)), r[0]);
}
TEST(AgradFwdLogInvLogit,FvarFvarVar_3rdDeriv) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<fvar<var> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 1.0;

  fvar<fvar<var> > a = log_inv_logit(x);

  AVEC p = createAVEC(x.val_.val_);
  VEC g;
  a.d_.d_.grad(p,g);
  EXPECT_FLOAT_EQ(0.057556793, g[0]);
}
struct log_inv_logit_fun {
  template <typename T0>
  inline T0
  operator()(const T0& arg1) const {
    return log_inv_logit(arg1);
  }
};

TEST(AgradFwdLogInvLogit,log_inv_logit_NaN) {
  log_inv_logit_fun log_inv_logit_;
  test_nan(log_inv_logit_,false);
}
