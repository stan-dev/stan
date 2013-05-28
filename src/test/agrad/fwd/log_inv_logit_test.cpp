#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log_inv_logit.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, log_inv_logit){
  using stan::agrad::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(0.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

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

TEST(AgradFvarVar, log_inv_logit) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<var> x;
  x.val_ = 0.5;
  x.d_ = 1.3;
  fvar<var> a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * exp(-0.5) / (1 + exp(-0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFvarFvar, log_inv_logit) {
  using stan::agrad::fvar;
  using stan::math::log_inv_logit;
  using std::exp;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > a = log_inv_logit(x);

  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  a = log_inv_logit(y);
  EXPECT_FLOAT_EQ(log_inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(exp(-0.5) / (1 + exp(-0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
