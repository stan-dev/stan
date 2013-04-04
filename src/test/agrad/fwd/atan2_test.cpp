#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, atan2) {
  using stan::agrad::fvar;
  using std::atan2;

  fvar<double> x(0.5);
  fvar<double> y(2.3);
  x.d_ = 1.0;
  y.d_ = 2.0;
  double w = 2.1;

  fvar<double> a = atan2(x, y);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.3), a.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.3 - 0.5 * 2.0) / (0.5 * 0.5 + 2.3 * 2.3), a.d_);

  fvar<double> b = atan2(w, x);
  EXPECT_FLOAT_EQ(atan2(2.1, 0.5), b.val_);
  EXPECT_FLOAT_EQ((-2.1 * 1.0) / (2.1 * 2.1 + 0.5 * 0.5), b.d_);

  fvar<double> c = atan2(x, w);
  EXPECT_FLOAT_EQ(atan2(0.5, 2.1), c.val_);
  EXPECT_FLOAT_EQ((1.0 * 2.1) / (0.5 * 0.5 + 2.1 * 2.1), c.d_);
}
