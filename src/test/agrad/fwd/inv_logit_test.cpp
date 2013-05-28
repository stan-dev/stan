#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/inv_logit.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, invLogit) {
  using stan::agrad::fvar;
  using stan::math::inv_logit;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = inv_logit(x);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(-1.2), b.val_);
  EXPECT_FLOAT_EQ(inv_logit(-1.2) * (1 - inv_logit(-1.2)), b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = inv_logit(z);
  EXPECT_FLOAT_EQ(inv_logit(1.5), c.val_);
  EXPECT_FLOAT_EQ(inv_logit(1.5) * (1 - inv_logit(1.5)), c.d_);
}

TEST(AgradFvarVar, inv_logit) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::inv_logit;

  fvar<var> x;
  x.val_ = 0.5;
  x.d_ = 1.3;
  fvar<var> a = inv_logit(x);

  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 * inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0, g[0]);
}

TEST(AgradFvarFvar, inv_logit) {
  using stan::agrad::fvar;
  using stan::math::inv_logit;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > a = inv_logit(x);

  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  a = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
