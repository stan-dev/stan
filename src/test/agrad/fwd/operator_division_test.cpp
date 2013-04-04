#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, operatorDivision){
  using stan::agrad::fvar;
  using std::isnan;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 / x2;

  EXPECT_FLOAT_EQ(0.5 / 0.4, a.val_);
  EXPECT_FLOAT_EQ((1.0 * 0.4 - 2.0 * 0.5) / (0.4 * 0.4), a.d_);

  fvar<double> b = -x1 / x2;
  EXPECT_FLOAT_EQ(-0.5 / 0.4, b.val_);
  EXPECT_FLOAT_EQ((-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), b.d_);

  fvar<double> c = -3 * x1 / x2;
  EXPECT_FLOAT_EQ(-3 * 0.5 / 0.4, c.val_);
  EXPECT_FLOAT_EQ(3 * (-1 * 0.4 + 2.0 * 0.5) / (0.4 * 0.4), c.d_);

  fvar<double> x3(0.5);
  x3.d_ = 1.0;
   double x4 = 2.0;

  fvar<double> e = x4 / x3;
  EXPECT_FLOAT_EQ(2 / 0.5, e.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0 / (0.5 * 0.5), e.d_);

  fvar<double> f = x3 / -2;
  EXPECT_FLOAT_EQ(0.5 / -2, f.val_);
  EXPECT_FLOAT_EQ(1.0 / -2, f.d_);

  fvar<double> x5(0.0);
  x5.d_ = 1.0;
  fvar<double> g = x3/x5;
  isnan(g.val_);
  isnan(g.d_);
}
