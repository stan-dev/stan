#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>

TEST(AgradFvar, cosh) {
  using stan::agrad::fvar;
  using std::sinh;
  using std::cosh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = cosh(x);
  EXPECT_FLOAT_EQ(cosh(0.5), a.val_);
  EXPECT_FLOAT_EQ(sinh(0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = cosh(y);
  EXPECT_FLOAT_EQ(cosh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(sinh(-1.2), b.d_);

  fvar<double> c = cosh(-x);
  EXPECT_FLOAT_EQ(cosh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-sinh(-0.5), c.d_);
}
