#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/inv_logit.hpp>

TEST(AgradFvar, invLogit) {
  using stan::agrad::fvar;
  using stan::math::inv_logit;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = inv_logit(x);
  EXPECT_FLOAT_EQ(inv_logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(inv_logit(0.5) * (1 - inv_logit(0.5)), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = inv_logit(y);
  EXPECT_FLOAT_EQ(inv_logit(-1.2), b.val_);
  EXPECT_FLOAT_EQ(inv_logit(-1.2) * (1 - inv_logit(-1.2)), b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = inv_logit(z);
  EXPECT_FLOAT_EQ(inv_logit(1.5), c.val_);
  EXPECT_FLOAT_EQ(inv_logit(1.5) * (1 - inv_logit(1.5)), c.d_);
}
