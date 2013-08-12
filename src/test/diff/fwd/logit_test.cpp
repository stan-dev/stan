#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/logit.hpp>

TEST(AgradFvar, logit) {
  using stan::agrad::fvar;
  using stan::math::logit;
  using std::isnan;

  fvar<double> x(0.5);
  x.d_ = 1.0;

  fvar<double> a = logit(x);
  EXPECT_FLOAT_EQ(logit(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (0.5 - 0.5 * 0.5), a.d_);

  fvar<double> y(-1.2);
  y.d_ = 1.0;

  fvar<double> b = logit(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> z(1.5);
  z.d_ = 1.0;

  fvar<double> c = logit(z);
  isnan(c.val_);
  isnan(c.d_);
}
