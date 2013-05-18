#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, operatorSubtraction){
  using stan::agrad::fvar;

  fvar<double> x1(0.5);
  fvar<double> x2(0.4);
  x1.d_ = 1.0;
  x2.d_ = 2.0;
  fvar<double> a = x1 - x2;
  EXPECT_FLOAT_EQ(0.5 - 0.4, a.val_);
  EXPECT_FLOAT_EQ(1.0 - 2.0, a.d_);

  fvar<double> b = -x1 - x2;
  EXPECT_FLOAT_EQ(-0.5 - 0.4, b.val_);
  EXPECT_FLOAT_EQ(-1 * 1.0 - 2.0, b.d_);

  fvar<double> c = 2 * x1 - -3 * x2;
  EXPECT_FLOAT_EQ(2 * 0.5 - -3 * 0.4, c.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - -3 * 2.0, c.d_);

  fvar<double> x3(0.5);
  fvar<double> x4(1.0);
  x3.d_ = 1.0;
  x4.d_ = 2.0;

  fvar<double> d = 2 * x3 - x4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 1 * 1.0, d.val_);
  EXPECT_FLOAT_EQ(2 * 1.0 - 1 * 2.0, d.d_);

  fvar<double> e = 2 * x3 - 4;
  EXPECT_FLOAT_EQ(2 * 0.5 - 4, e.val_);
  EXPECT_FLOAT_EQ(2 * 1.0, e.d_);

  fvar<double> f = 5 - 2 * x3;
  EXPECT_FLOAT_EQ(5 - 2 * 0.5, f.val_);
  EXPECT_FLOAT_EQ(-2 * 1.0, f.d_);
}
