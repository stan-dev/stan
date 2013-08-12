#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/cbrt.hpp>

TEST(AgradFvar, cbrt) {
  using stan::agrad::fvar;
  using boost::math::cbrt;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0; //derivatives w.r.t. x
  fvar<double> a = cbrt(x);

  EXPECT_FLOAT_EQ(cbrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (3 * pow(0.5, 2.0 / 3.0)), a.d_);

  fvar<double> b = 3 * cbrt(x) + x;
  EXPECT_FLOAT_EQ(3 * cbrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (3 * pow(0.5, 2.0 / 3.0)) + 1, b.d_);

  fvar<double> c = -cbrt(x) + 5;
  EXPECT_FLOAT_EQ(-cbrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (3 * pow(0.5, 2.0 / 3.0)), c.d_);

  fvar<double> d = -3 * cbrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (3 * pow(0.5, 2.0 / 3.0)) + 5, d.d_);

  fvar<double> e = -3 * cbrt(-x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cbrt(-0.5) + 5 * 0.5, e.val_);
  EXPECT_FLOAT_EQ(3 / (3 * cbrt(-0.5) * cbrt(-0.5)) + 5, e.d_);

  fvar<double> y(0.0);
  y.d_ = 1.0;
  fvar<double> f = cbrt(y);
  EXPECT_FLOAT_EQ(cbrt(0.0), f.val_);
  isnan(f.d_);
}
