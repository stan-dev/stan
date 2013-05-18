#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/binary_log_loss.hpp>

TEST(AgradFvar, binary_log_loss) {
  using stan::agrad::fvar;
  using stan::math::binary_log_loss;
  using std::log;
  using std::isnan;

  fvar<double> w(0.0);
  fvar<double> x(1.0);
  w.d_ = 1.0;
  x.d_ = 2.0;
  fvar<double> y(0.4);
  y.d_ = 3.0;

  double p = 0.0;
  double r = 0.4;

  fvar<double> a = binary_log_loss(w, y);
  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), a.val_);
  EXPECT_FLOAT_EQ(-1.0 * log(0.4) + 1.0 * log(1 - 0.4) + 3.0 / 0.6, a.d_);

  fvar<double> b = binary_log_loss(x, y);
  EXPECT_FLOAT_EQ(binary_log_loss(1.0, 0.4), b.val_);
  EXPECT_FLOAT_EQ(-2.0 * log(0.4) + 2.0 * log(1 - 0.4) - 3.0 * 1.0 / 0.4, b.d_);

  fvar<double> c = binary_log_loss(p, y);
  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), c.val_);
  EXPECT_FLOAT_EQ(3.0 * 1 / (1 - 0.4), c.d_);

  fvar<double> d = binary_log_loss(w, r);
  EXPECT_FLOAT_EQ(binary_log_loss(0.0, 0.4), d.val_);
  EXPECT_FLOAT_EQ(-1.0 * log(0.4) + 1.0 * log(0.6), d.d_);

  fvar<double> e = binary_log_loss(y, r);
  isnan(e.val_);
  isnan(e.d_);

  double s = 1.2;

  fvar<double> f = binary_log_loss(s, y);
  isnan(f.val_);
  isnan(f.d_);
}
