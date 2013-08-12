#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/asinh.hpp>

TEST(AgradFvar, asinh) {
  using stan::agrad::fvar;
  using boost::math::asinh;
  using std::sqrt;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = asinh(x);
  EXPECT_FLOAT_EQ(asinh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (0.5) * (0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = asinh(y);
  EXPECT_FLOAT_EQ(asinh(-1.2), b.val_);
  EXPECT_FLOAT_EQ(1 / sqrt(1 + (-1.2) * (-1.2)), b.d_);

  fvar<double> c = asinh(-x);
  EXPECT_FLOAT_EQ(asinh(-0.5), c.val_);
  EXPECT_FLOAT_EQ(-1 / sqrt(1 + (-0.5) * (-0.5)), c.d_);
}
