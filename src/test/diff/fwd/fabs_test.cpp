#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/constants.hpp>

TEST(AgradFvar, fabs) {
  using stan::agrad::fvar;
  using std::fabs;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fabs(x);
  EXPECT_FLOAT_EQ(fabs(2), a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fabs(-x);
  EXPECT_FLOAT_EQ(fabs(-2), b.val_);
  EXPECT_FLOAT_EQ(1.0, b.d_);

  fvar<double> c = fabs(y);
  EXPECT_FLOAT_EQ(fabs(-3), c.val_);
  EXPECT_FLOAT_EQ(-2.0, c.d_);

  fvar<double> d = fabs(x * 2);
  EXPECT_FLOAT_EQ(fabs(2 * 2), d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, d.d_);

  fvar<double> e = fabs(y + 4);
  EXPECT_FLOAT_EQ(fabs(-3.0 + 4), e.val_);
  EXPECT_FLOAT_EQ(2.0, e.d_);

  fvar<double> f = fabs(x - 2);
  EXPECT_FLOAT_EQ(fabs(2.0 - 2), f.val_);
  isnan(f.d_);
 }
