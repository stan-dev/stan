#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, floor) {
  using stan::agrad::fvar;
  using std::floor;

  fvar<double> x(0.5);
  fvar<double> y(2.0);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = floor(x);
  EXPECT_FLOAT_EQ(floor(0.5), a.val_);
  EXPECT_FLOAT_EQ(0, a.d_);

  fvar<double> b = floor(y);
  EXPECT_FLOAT_EQ(floor(2.0), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = floor(2 * x);
  EXPECT_FLOAT_EQ(floor(2 * 0.5), c.val_);
   EXPECT_FLOAT_EQ(0.0, c.d_);
}
