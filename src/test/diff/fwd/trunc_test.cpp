#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, trunc) {
  using stan::agrad::fvar;
  using boost::math::trunc;

  fvar<double> x(0.5);
  fvar<double> y(2.4);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = trunc(x);
  EXPECT_FLOAT_EQ(trunc(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = trunc(y);
  EXPECT_FLOAT_EQ(trunc(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = trunc(2 * x);
  EXPECT_FLOAT_EQ(trunc(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);
}
