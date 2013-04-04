#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

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
