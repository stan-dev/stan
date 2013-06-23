#include <gtest/gtest.h>
#include <stan/agrad/fvar.hpp>
#include <stan/math/functions/log1m.hpp>

TEST(AgradFvar, log1m){
  using stan::agrad::fvar;
  using stan::math::log1m;
  using std::isnan;

  fvar<double> x(0.5);
  fvar<double> y(1.0);
  fvar<double> z(2.0);
  x.d_ = 1.0;
  y.d_ = 2.0;
  z.d_ = 3.0;

  fvar<double> a = log1m(x);
  EXPECT_FLOAT_EQ(log1m(0.5), a.val_);
  EXPECT_FLOAT_EQ(-1 / (1 - 0.5), a.d_);

  fvar<double> b = log1m(y);
  isnan(b.val_);
  isnan(b.d_);

  fvar<double> c = log1m(z);
  isnan(c.val_);
  isnan(c.d_);
}
