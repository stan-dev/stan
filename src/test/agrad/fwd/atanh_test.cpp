#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <boost/math/special_functions/atanh.hpp>

TEST(AgradFvar, atanh) {
  using stan::agrad::fvar;
  using boost::math::atanh;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = atanh(x);
  EXPECT_FLOAT_EQ(atanh(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.5 * 0.5), a.d_);

  fvar<double> y(-0.9);
  y.d_ = 1.0;

  fvar<double> b = atanh(y);
  EXPECT_FLOAT_EQ(atanh(-0.9), b.val_);
  EXPECT_FLOAT_EQ(1 / (1 - 0.9 * 0.9), b.d_);
}
