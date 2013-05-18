#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/round.hpp>

TEST(AgradFvar, round) {
  using stan::agrad::fvar;
  using boost::math::round;

  fvar<double> x(0.5);
  fvar<double> y(2.4);
  y.d_ = 2.0;
  x.d_ = 1.0;

  fvar<double> a = round(x);
  EXPECT_FLOAT_EQ(round(0.5), a.val_);
  EXPECT_FLOAT_EQ(0.0, a.d_);

  fvar<double> b = round(y);
  EXPECT_FLOAT_EQ(round(2.4), b.val_);
  EXPECT_FLOAT_EQ(0.0, b.d_);

  fvar<double> c = round(2 * x);
  EXPECT_FLOAT_EQ(round(2 * 0.5), c.val_);
  EXPECT_FLOAT_EQ(0.0, c.d_);

  fvar<double> z(1.25);
  z.d_ = 1.0;

  fvar<double> d = round(2 * z);
  EXPECT_FLOAT_EQ(round(2 * 1.25), d.val_);
   EXPECT_FLOAT_EQ(0.0, d.d_);
}
