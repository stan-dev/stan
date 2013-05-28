#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/agrad/var.hpp>
#include <test/agrad/util.hpp>

TEST(AgradFvar, abs) {
  using stan::agrad::fvar;
  using std::abs;
  using std::isnan;

  fvar<int> x(2);
  fvar<int> y(-3);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<int> a = abs(x);
  EXPECT_FLOAT_EQ(abs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<int> b = abs(-x);
  EXPECT_FLOAT_EQ(abs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<int> c = abs(y);
  EXPECT_FLOAT_EQ(abs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = abs(2 * x);
  EXPECT_FLOAT_EQ(abs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = abs(y + 4);
  EXPECT_FLOAT_EQ(abs(-3 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = abs(x - 2);
  EXPECT_FLOAT_EQ(abs(2 - 2), f.val_);
  isnan(f.d_);
 }

TEST(AgradFvarVar, abs) {
  using stan::agrad::fvar;
  using stan::agrad::var;
  using std::abs;

  fvar<var> x;
  x.val_ = 2.0;
  x.d_ = 1.0;
  fvar<var> a = abs(x);

  EXPECT_FLOAT_EQ(2.0, a.val_.val());
  EXPECT_FLOAT_EQ(1.0, a.d_.val());

  AVEC y = createAVEC(x.val_);
  VEC g;
  a.val_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);

  y = createAVEC(x.d_);
  a.d_.grad(y,g);
  EXPECT_FLOAT_EQ(1.0, g[0]);
}

TEST(AgradFvarFvar, abs) {
  using stan::agrad::fvar;
  using std::abs;

  fvar<fvar<double> > x;
  x.val_.val_ = 4.0;
  x.val_.d_ = 1.0;

  fvar<fvar<double> > a = abs(x);

  EXPECT_FLOAT_EQ(4.0, a.val_.val_);
  EXPECT_FLOAT_EQ(1.0, a.val_.d_);
  EXPECT_FLOAT_EQ(0.0, a.d_.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_.d_);
}
