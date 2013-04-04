#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, pow) {
  using stan::agrad::fvar;
  using std::pow;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;
  double y = 5.0;

  fvar<double> a = pow(x, y);
  EXPECT_FLOAT_EQ(pow(0.5, 5.0), a.val_);
  EXPECT_FLOAT_EQ(5.0 * pow(0.5, 5.0 - 1.0), a.d_);

  fvar<double> b = pow(y, x);
  EXPECT_FLOAT_EQ(pow(5.0, 0.5), b.val_);
  EXPECT_FLOAT_EQ(log(5.0) * pow(5.0, 0.5), b.d_);

  fvar<double> z(1.2);
  z.d_ = 2.0;
  fvar<double> c = pow(x, z);
  EXPECT_FLOAT_EQ(pow(0.5, 1.2), c.val_);
  EXPECT_FLOAT_EQ((2.0 * log(0.5) + 1.2 * 1.0 / 0.5) * pow(0.5, 1.2), c.d_);

  fvar<double> w(-0.4);
  w.d_ = 1.0;
  fvar<double> d = pow(w, x);
  isnan(d.val_);
  isnan(d.d_);
}
