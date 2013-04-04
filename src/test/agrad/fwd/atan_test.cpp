#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, atan) {
  using stan::agrad::fvar;
  using std::atan;

  fvar<double> x(0.5);
  x.d_ = 1.0;   // derivatives w.r.t. x
  
  fvar<double> a = atan(x);
  EXPECT_FLOAT_EQ(atan(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5 * 0.5), a.d_);

  fvar<double> b = 2 * atan(x) + 4;
  EXPECT_FLOAT_EQ(2 * atan(0.5) + 4, b.val_);
  EXPECT_FLOAT_EQ(2 / (1 + 0.5 * 0.5), b.d_);

  fvar<double> c = -atan(x) + 5;
  EXPECT_FLOAT_EQ(-atan(0.5) + 5, c.val_);
  EXPECT_FLOAT_EQ(-1 / (1 + 0.5 * 0.5), c.d_);

  fvar<double> d = -3 * atan(x) + 5 * x;
  EXPECT_FLOAT_EQ(-3 * atan(0.5) + 5 * 0.5, d.val_);
  EXPECT_FLOAT_EQ(-3 / (1 + 0.5 * 0.5) + 5, d.d_);
}
