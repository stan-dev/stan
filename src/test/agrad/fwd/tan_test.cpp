#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, tan) {
  using stan::agrad::fvar;
  using std::tan;
  using std::cos;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = tan(x);
  EXPECT_FLOAT_EQ(tan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.5) * cos(0.5)), a.d_);

  fvar<double> b = 2 * tan(x) + 4;
  EXPECT_FLOAT_EQ(2 * tan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (cos(0.5) * cos(0.5)), b.d_);

  fvar<double> c = -tan(x) + 5;
  EXPECT_FLOAT_EQ(-tan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (cos(0.5) * cos(0.5)), c.d_);

  fvar<double> d = -3 * tan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * tan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (cos(0.5) * cos(0.5)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = tan(y);
  EXPECT_FLOAT_EQ(tan(-0.5), e.val_);
  EXPECT_FLOAT_EQ(1 / (cos(-0.5) * cos(-0.5)), e.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> f = tan(z);
  EXPECT_FLOAT_EQ(tan(0.0), f.val_);
  EXPECT_FLOAT_EQ(1 / (cos(0.0) * cos(0.0)), f.d_);
}
