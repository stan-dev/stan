#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, log) {
  using stan::agrad::fvar;
  using std::log;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = log(x);
  EXPECT_FLOAT_EQ(log(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / 0.5, a.d_);

  fvar<double> b = 2 * log(x) + 4;
  EXPECT_FLOAT_EQ(2 * log(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / 0.5, b.d_);

  fvar<double> c = -log(x) + 5;
  EXPECT_FLOAT_EQ(-log(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / 0.5, c.d_);

  fvar<double> d = -3 * log(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * log(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / 0.5 + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = log(y);
  isnan(e.val_);
  isnan(e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = log(z);
  isnan(f.val_);
  isnan(f.d_);
}
