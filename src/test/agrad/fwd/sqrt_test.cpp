#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, sqrt) {
  using stan::agrad::fvar;
  using std::sqrt;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0; //derivatives w.r.t. x
  fvar<double> a = sqrt(x);

  EXPECT_FLOAT_EQ(sqrt(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (2 * sqrt(0.5)), a.d_);

  fvar<double> b = 3 * sqrt(x) + x;
  EXPECT_FLOAT_EQ(3 * sqrt(0.5) + 0.5, b.val_);
  EXPECT_FLOAT_EQ(3 / (2 * sqrt(0.5)) + 1, b.d_);

  fvar<double> c = -sqrt(x) + 5;
  EXPECT_FLOAT_EQ(-sqrt(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), c.d_);

  fvar<double> d = -3 * sqrt(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * sqrt(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (2 * sqrt(0.5)) + 5, d.d_);

  fvar<double> y(-0.5);
  y.d_ = 1.0;
  fvar<double> e = sqrt(-y);
  EXPECT_FLOAT_EQ(sqrt(0.5), e.val_);
  EXPECT_FLOAT_EQ(-1 / (2 * sqrt(0.5)), e.d_); 

  fvar<double> f = sqrt(y);
  isnan(f.val_);
  isnan(f.d_);

  fvar<double> z(0.0);
  z.d_ = 1.0;
  fvar<double> g = sqrt(z);
  EXPECT_FLOAT_EQ(sqrt(0.0), g.val_);
  isnan(g.d_); 
}
