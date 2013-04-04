#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, fmax) {
  using stan::agrad::fvar;
  using std::isnan;

  fvar<double> x(2.0);
  fvar<double> y(-3.0);
  x.d_ = 1.0;
  y.d_ = 2.0;

  fvar<double> a = fmax(x, y);
  EXPECT_FLOAT_EQ(2.0, a.val_);
  EXPECT_FLOAT_EQ(1.0, a.d_);

  fvar<double> b = fmax(2 * x, y);
  EXPECT_FLOAT_EQ(4.0, b.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, b.d_);

  fvar<double> c = fmax(y, x);
  EXPECT_FLOAT_EQ(2.0, c.val_);
  EXPECT_FLOAT_EQ(1.0, c.d_);

  fvar<double> d = fmax(x, x);
  EXPECT_FLOAT_EQ(2.0, d.val_);
  isnan(d.d_);

  double z = 1.0;

  fvar<double> e = fmax(x, z);
  EXPECT_FLOAT_EQ(2.0, e.val_);
  EXPECT_FLOAT_EQ(1.0, e.d_);

  fvar<double> f = fmax(z, x);
  EXPECT_FLOAT_EQ(2.0, f.val_);
  EXPECT_FLOAT_EQ(1.0, f.d_);
 }
