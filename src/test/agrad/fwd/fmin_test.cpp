#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, fmin) {
  using stan::agrad::fvar;
  using stan::agrad::fmin;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmin(x, y);
  EXPECT_FLOAT_EQ(-3.0, a.val_);
  EXPECT_FLOAT_EQ(2.0, a.d_);

  fvar<double> b = fmin(2 * x, y);
  EXPECT_FLOAT_EQ(-3.0, b.val_);
  EXPECT_FLOAT_EQ(2.0, b.d_);

  fvar<double> c = fmin(y, x);
  EXPECT_FLOAT_EQ(-3.0, c.val_);
  EXPECT_FLOAT_EQ(2.0, c.d_);

  fvar<double> d = fmin(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmin(x, z);
  EXPECT_FLOAT_EQ(1.0, e.val_);
  EXPECT_FLOAT_EQ(0.0, e.d_);

  fvar<double> f = fmin(z, x);
  EXPECT_FLOAT_EQ(1.0, f.val_);
  EXPECT_FLOAT_EQ(0.0, f.d_);
 }

TEST(AgradFvarVar, fmin) {
  using stan::agrad::fvar;
  using stan::agrad::var;

  fvar<var> x;
  x.val_ = 2.5;
  x.d_ = 1.3;

  fvar<var> z;
  z.val_ = 1.5;
  z.d_ = 1.0;
  fvar<var> a = fmin(x,z);

  EXPECT_FLOAT_EQ(fmin(2.5,1.5), a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(0,g[0]);
  std::isnan(g[1]);
}

TEST(AgradFvarFvar, fmin) {
  using stan::agrad::fvar;

  fvar<fvar<double> > x;
  x.val_.val_ = 2.5;
  x.val_.d_ = 1.0;
  x.d_.val_ = 0.0;
  x.d_.d_ = 0.0;

  fvar<fvar<double> > y;
  y.val_.val_ = 1.5;
  y.val_.d_ = 0.0;
  y.d_.val_ = 1.0;
  y.d_.d_ = 0.0;

  fvar<fvar<double> > a = fmin(x,y);

  EXPECT_FLOAT_EQ(fmin(2.5,1.5), a.val_.val_);
  EXPECT_FLOAT_EQ(0, a.val_.d_);
  EXPECT_FLOAT_EQ(1, a.d_.val_);
  EXPECT_FLOAT_EQ(0, a.d_.d_);
}
