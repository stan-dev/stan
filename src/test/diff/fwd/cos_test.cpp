#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, cos) {
  using stan::agrad::fvar;
  using std::sin;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = cos(x);
  EXPECT_FLOAT_EQ(cos(0.5), a.val_);
  EXPECT_FLOAT_EQ(-sin(0.5), a.d_);

  fvar<double> b = 2 * cos(x) + 4;
  EXPECT_FLOAT_EQ(2 * cos(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 * -sin(0.5), b.d_);

  fvar<double> c = -cos(x) + 5;
  EXPECT_FLOAT_EQ(-cos(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(sin(0.5), c.d_);

  fvar<double> d = -3 * cos(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * cos(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 * -sin(0.5) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = cos(y);
  EXPECT_FLOAT_EQ(cos(-0.5), e.val_);
  EXPECT_FLOAT_EQ(-sin(-0.5), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = cos(z);
  EXPECT_FLOAT_EQ(cos(0.0), f.val_);
  EXPECT_FLOAT_EQ(-sin(0.0), f.d_);
}
