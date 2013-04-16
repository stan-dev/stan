#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/inv_cloglog.hpp>

TEST(AgradFvar, invCLogLog) {
  using stan::agrad::fvar;
  using stan::math::inv_cloglog;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = inv_cloglog(x);
  EXPECT_FLOAT_EQ(inv_cloglog(0.5), a.val_);
  EXPECT_FLOAT_EQ(exp(0.5 -exp(0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = inv_cloglog(y);
  EXPECT_FLOAT_EQ(inv_cloglog(-1.2), b.val_);
  EXPECT_FLOAT_EQ(exp(-1.2 -exp(-1.2)), b.d_);

  fvar<double> z(1.5);
  z.d_ = 2.0;

  fvar<double> c = inv_cloglog(z);
  EXPECT_FLOAT_EQ(inv_cloglog(1.5), c.val_);
  EXPECT_FLOAT_EQ(2.0 * exp(1.5 -exp(1.5)), c.d_);
}
