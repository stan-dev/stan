#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log1p.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, log1p) {
  using stan::agrad::fvar;
  using stan::math::log1p;
  using std::isnan;

  fvar<double> x(0.5,1.0);
  fvar<double> y(-1.0,2.0);
  fvar<double> z(-2.0,3.0);

  fvar<double> a = log1p(x);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5), a.d_);

  fvar<double> b = log1p(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1p(z);
  isnan(c.val_);
  isnan(c.d_);
}

TEST(AgradFvarVar, log1p) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using stan::math::log1p;

  fvar<var> x(0.5,1.3);
  fvar<var> a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.3 / (1 + 0.5), a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1 / (1.5), g[0]);
}

TEST(AgradFvarFvar, log1p) {
  using stan::agrad::fvar;
  using stan::math::log1p;

  fvar<fvar<double> > x;
  x.val_.val_ = 0.5;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = log1p(x);

  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(1 / (1.5), a.val_.d_);
  EXPECT_FLOAT_EQ(0, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);

  fvar<fvar<double> > y;
  y.val_.val_ = 0.5;
  y.d_.val_ = 1.0;

  a = log1p(y);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1 / (1.5), a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}

