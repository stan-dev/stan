#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log1p.hpp>

TEST(AgradFvar, log1p){
  using stan::agrad::fvar;
  using stan::math::log1p;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(-1.0);
  fvar<double> z(-2.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1p(x);
  EXPECT_FLOAT_EQ(log1p(0.5), a.val_);
  EXPECT_FLOAT_EQ(1 / (1 + 0.5), a.d_);

  fvar<double> b = log1p(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1p(z);
  isnan(c.val_);
  isnan(c.d_);
}
