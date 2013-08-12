#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/multiply_log.hpp>

TEST(AgradFvar,multiply_log) {
  using stan::agrad::fvar;
  using std::isnan;
  using std::log;
  using stan::math::multiply_log;

  fvar<double> x(0.5);
  fvar<double> y(1.2);
  fvar<double> z(-0.4);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  double w = 0.0;
  double v = 1.3;

  fvar<double> a = multiply_log(x, y);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.2), a.val_);
  EXPECT_FLOAT_EQ(1.0 * log(1.2) + 0.5 * 2.0 / 1.2, a.d_);

  fvar<double> b = multiply_log(x,z);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = multiply_log(x, v);
  EXPECT_FLOAT_EQ(multiply_log(0.5, 1.3), c.val_);
  EXPECT_FLOAT_EQ(log(1.3), c.d_);

  fvar<double> d = multiply_log(v, x);
  EXPECT_FLOAT_EQ(multiply_log(1.3, 0.5), d.val_);
  EXPECT_FLOAT_EQ(1.3 * 1.0 / 0.5, d.d_);

  fvar<double> e = multiply_log(x, w);
  isnan(e.val_);
  isnan(e.d_);
}
